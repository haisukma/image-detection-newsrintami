from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import shutil
import os

app = FastAPI()

model = YOLO("best1.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobbj(file.file, buffer)

    results = model(file_path)

    detections = []

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": model.names[cls_id],
            "confidence": round(conf, 4)
        })

    return {
        "filename": file.filename,
        "detections": detections
    }