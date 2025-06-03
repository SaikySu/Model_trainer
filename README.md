# Background Removal + Vectorization API with Hailo-8

## Introduction
An API built with FastAPI to **process input images**, including:
1. **Background removal** using a segmentation model.
2. **Feature extraction** (embedding vector).
3. **Saving extracted vectors** to a JSON-based database.
4. Returning the processed results via REST API.

---

## Technologies Used

| Component           | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| **FastAPI**         | A fast, scalable web framework for building APIs                       |
| **Hailo-8 SDK**     | Uses `hailo_platform` to run `.hef` models on the Hailo-8 AI device    |
| **Pillow, NumPy**   | Image processing (loading, resizing, masking, generating RGBA images) |
| **Uvicorn**         | ASGI server for running FastAPI                                        |

---

## Models Used

| Model Name                    | Purpose                              | Format |
|------------------------------|--------------------------------------|--------|
| `Your deeplab_model.hef`| Semantic segmentation - background removal | `.hef` |
| `Your_resnet_model.hef`           | Extract embedding vector from image  | `.hef` |

> Both models are compiled and executed on **Hailo-8** using the `hailo_platform`.

---

## Main Endpoints

| Endpoint                            | Description |
|-------------------------------------|-------------|
| `POST /process_image/`             | Upload image → remove background → vectorize → save to JSON |
| `GET /get_processed_image/{file}`  | Return processed image (PNG with transparent background)     |
| `GET /get_vector_db/`              | Get all stored vectors from the database                     |
| `GET /`                            | API overview and available endpoints                         |

---

## Example Output (`POST /process_image/`)

```json
{
  "message": "Image processed successfully",
  "original_filename": "sample.jpg",
  "processed_filename": "sample_no_bg.png",
  "bg_removed_path": "processed/sample_no_bg.png",
  "bg_removed_url": "/processed/sample_no_bg.png",
  "vector_shape": [1001],
  "vector_preview": [0.02, 0.12, ..., 0.01]
}
```

## ⚙️ System Requirements
- Ubuntu + Hailo SDK 4.x
- Required packages:
    - hailo_platform
    - fastapi, uvicorn
    - numpy, Pillow

- Quick install: 
```bash
pip install fastapi uvicorn numpy pillow
``` 

## Note
- You can use the command `hailortcli` scan to check connected Hailo-8 devices.
- The vector database is saved in `vector_db.json`. You may upgrade to SQLite or FAISS for larger-scale systems.
- Every uploaded image generates a corresponding background-removed .png and an embedding vector that is saved.
