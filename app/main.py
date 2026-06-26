from pathlib import Path
import os
import shutil
import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from app.detector import model
from app.image_utils import crop_bbox

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

@app.post("/predict")
async def predict(
    request: Request,
    expected_item: str = Form(...),
    file: UploadFile = File(...)
):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(
        file_path,
        conf=0.25
    )

    img = cv2.imread(file_path)

    detections = []

    target_detection = None

    for box in results[0].boxes:

        conf = float(box.conf[0])

        if conf < 0.25:
            continue

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        bbox = [x1, y1, x2, y2]

        label = f"{class_name} {conf:.2f}"

        detections.append({
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": bbox
        })

        if class_name.lower() == expected_item.lower():

            if (
                target_detection is None
                or conf > target_detection["confidence"]
            ):

                target_detection = {
                    "class": class_name,
                    "confidence": round(conf, 4),
                    "bbox": bbox
                }
        
        color = (255, 0, 0)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        font_scale = 0.8
        font_thickness = 2
        padding = 5

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        text_x = x1
        text_y = max(text_h + 10, y1 - 10)

        cv2.rectangle(
            img,
            (text_x, text_y - text_h - padding),
            (text_x + text_w + padding * 2, text_y + padding),
            color,
            -1
        )

        cv2.putText(
            img,
            label,
            (text_x + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

    if target_detection is None:

        return {
            "success": False,
            "message": f"{expected_item} tidak ditemukan pada gambar.",
            "detections": detections
        }

    output_filename = f"result_{file.filename}"

    output_path = f"uploads/{output_filename}"

    cv2.imwrite(
        output_path,
        img
    )

    return {
        "success": True,
        "filename": file.filename,
        "image_url": str(request.base_url) + f"uploads/{output_filename}",
        "target_item": target_detection,
        "detections": detections
    }