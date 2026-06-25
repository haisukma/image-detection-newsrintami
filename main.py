from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import shutil
from pathlib import Path
import cv2
import os

app = FastAPI()

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "model" / "best4.pt"

model = YOLO(str(MODEL_PATH))

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

os.makedirs(
    "uploads",
    exist_ok=True
)

# CLASS_COLORS = {
#     "aksesoris cold": (72, 101, 241),
#     "aksesoris hot": (76, 175, 80),
#     "arcing horn": (255, 167, 38),
#     "bracing": (38, 198, 218),
#     "isolator": (229, 57, 53),
#     "jumper": (141, 110, 99),
#     "pondasi": (66, 133, 244),
# }

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # results = model(file_path)

    # annotated_frame = results[0].plot()

    results = model(
        file_path,
        conf=0.55
    )

    img = cv2.imread(file_path)

    detections = []

    for box in results[0].boxes:

        conf = float(box.conf[0])

        if conf < 0.55:
            continue

        cls_id = int(box.cls[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        label = f"{model.names[cls_id]} {conf:.2f}"

        color = (255, 0, 0)

        # class_name = model.names[cls_id]

        # color = CLASS_COLORS.get(
        #     class_name,
        #     (255, 255, 255)
        # )

        detections.append({
            "class": label,
            "confidence": round(conf, 4)
        })

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        font_scale = 1
        font_thickness = 2
        padding = 6

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        text_x = x1
        text_y = y1 - 10

        if text_y - th < 0:

            text_x = x2 + 10
            text_y = y1 + th

        img_h, img_w = img.shape[:2]

        if text_x + tw > img_w:

            text_x = img_w - tw - 10

        cv2.rectangle(
            img,
            (text_x, text_y - th - padding),
            (text_x + tw + padding*2, text_y + padding),
            color,
            -1
        )

        cv2.putText(
            img,
            label,
            (text_x + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255,255,255),
            font_thickness
        )

    annotated_frame = img

    output_filename = f"result_{file.filename}"

    output_path = f"uploads/{output_filename}"

    cv2.imwrite(
        output_path,
        annotated_frame
    )

    return {
        "filename": file.filename,
        "image_url": str(request.base_url) + f"uploads/{output_filename}",
        "detections": detections
    }