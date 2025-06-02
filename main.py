import os
import json
import shutil
import time
from uuid import uuid4
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

from hailo_platform import (
    HEF, VDevice, InputVStreamParams, OutputVStreamParams, InferVStreams,
    ConfigureParams, FormatType, HailoStreamInterface
)

# ---- CONFIGURATION ----
DEEPLAB_HEF_PATH = "deeplab_v3_mobilenet_v2.hef"
RESNET_HEF_PATH = "resnet_v1_50.hef"
VECTOR_JSON_PATH = "vector_db.json"
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# ---- FASTAPI ----
app = FastAPI(title="Background Removal + Vectorization API")

# Mount processed folder for static file serving
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.mount("/processed", StaticFiles(directory=PROCESSED_FOLDER), name="processed")

# Create hailo device, you can check hailo devices with: "hailortcli scan"
def create_device():
    try:
        device = VDevice()
        return device
    except Exception as e:
        raise RuntimeError(f"Hailo device not found or busy: {str(e)}")

# Use deeplab_v3_mobilenet_v2 (Segmention) to remove backgroud
def remove_background(img_path: str) -> str:
    # Create device + model
    device = create_device()
    hef = HEF(DEEPLAB_HEF_PATH)
    network_group = device.configure(hef)[0]

    # Input/output from .HEF model
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    height, width, channels = input_info.shape

    # Get shape info form HEF 
    input_vstreams_params = InputVStreamParams.make(network_group)
    output_vstreams_params = OutputVStreamParams.make(network_group)

    # Convert image to fit input data
    img = Image.open(img_path).convert("RGB")
    ori_size = img.size
    img_resized = img.resize((width, height), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32)
    img_np = np.expand_dims(img_np, axis=0)
    input_data = {input_info.name: img_np}

    # Run model and take input
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group.create_params()):
            infer_results = infer_pipeline.infer(input_data)
            output = infer_results[output_info.name]

    # Create RGBA with none backgroud 
    mask = np.argmax(output, axis=-1)[0]
    mask_img = Image.fromarray((mask != 0).astype(np.uint8) * 255)
    mask_img = mask_img.resize(ori_size, Image.NEAREST)
    mask_np = np.array(mask_img)

    original_np = np.array(img)
    alpha_channel = (mask_np > 0).astype(np.uint8) * 255
    result_rgba = np.dstack([original_np, alpha_channel])
    result_img = Image.fromarray(result_rgba)

    # Save image without backgroud
    basename = os.path.splitext(os.path.basename(img_path))[0]
    bg_removed_name = f"{basename}_no_bg.png"  
    bg_removed_path = os.path.join(PROCESSED_FOLDER, bg_removed_name)
    result_img.save(bg_removed_path)
    return bg_removed_path

# Resize image image to Tensort for Resnet and add batch demension (1, H, W, C)
def preprocess_image_for_vector(image_path: str, resnet_shape):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(resnet_shape)
    img_np = np.array(image).astype(np.float32)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def normalize_vector(v):
    return v / (np.linalg.norm(v) + 1e-8)

# Vectorize image 
def vectorize_image(image_path: str) -> np.ndarray:
    # Create Device
    device = create_device()
    hef = HEF(RESNET_HEF_PATH)
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_group = device.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    height, width, channels = input_info.shape

    input_vstreams_params = InputVStreamParams.make(
        network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(
        network_group, quantized=True, format_type=FormatType.UINT8)

    # Image preprocessing
    img_np = preprocess_image_for_vector(image_path, (width, height))
    input_data = {input_info.name: img_np}

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            result = infer_pipeline.infer(input_data)
            embedding = result[output_info.name][0].astype(np.float32)
            normalized = normalize_vector(embedding)
    return normalized

# Save vector convert from image to json file
def save_vector_to_json(vector, filename, original_filename, bg_removed_path):
    entry = {
        "original_filename": original_filename,
        "processed_filename": filename,
        "bg_removed_path": bg_removed_path,
        "vector_normalized": vector.tolist(),
        "vector_shape": list(vector.shape)
    }
    if os.path.exists(VECTOR_JSON_PATH):
        with open(VECTOR_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(VECTOR_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, jpeg, png)")
    try:
        # Dont read file.file.read() many time. UploadFile is Like-like, so just need read 1 time
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)         
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run rmbg funtion
        bg_removed_path = remove_background(img_path)

        # Convert img to vector without backgroud 
        vector = vectorize_image(bg_removed_path)

        # Save data to json file
        processed_filename = os.path.basename(bg_removed_path)
        save_vector_to_json(vector, processed_filename, file.filename, bg_removed_path)

        return JSONResponse(status_code=200, content={
            "message": "Xử lý ảnh thành công",
            "original_filename": file.filename,
            "processed_filename": processed_filename,
            "bg_removed_path": bg_removed_path,
            "bg_removed_url": f"/processed/{processed_filename}",
            "vector_shape": list(vector.shape),
            "vector_preview": vector[:10].tolist(),  # Show top 10 element  
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/get_processed_image/{filename}")
async def get_processed_image(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
    return FileResponse(file_path, media_type="image/png")

@app.get("/get_vector_db/")
async def get_vector_db():
    if not os.path.exists(VECTOR_JSON_PATH):
        return JSONResponse(content={"message": "Database trống", "data": []})
    with open(VECTOR_JSON_PATH, "r") as f:
        data = json.load(f)
    return JSONResponse(content={
        "data": data
    })

@app.get("/")
async def root():
    return JSONResponse(content={
        "message": "Background Removal + Vectorization API",
        "endpoints": {
            "POST /process_image/": "Upload và xử lý ảnh (tách nền + vectorize)",
            "GET /get_processed_image/{filename}": "Lấy ảnh đã tách nền",
            "GET /get_vector_db/": "Xem toàn bộ vector database"
        }
    })

if __name__ == "__main__":
    uvicorn.run("combined_api:app", host="0.0.0.0", port=8000, reload=True)