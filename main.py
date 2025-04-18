from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from ultralytics import YOLO
import io

app = FastAPI()

# Load YOLO model (adjust path and version as needed)
model = YOLO("model/best.pt")  # can be yolov8.pt or yolov12.pt

@app.get("/")
def root():
    return {"message": "Palm Oil Ripeness Detection API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = model(image)
    boxes = results[0].boxes
    predictions = []
    for box in boxes:
        label = int(box.cls[0])
        conf = float(box.conf[0])
        predictions.append({"class_id": label, "confidence": round(conf, 3)})

    return JSONResponse(content={"predictions": predictions})
