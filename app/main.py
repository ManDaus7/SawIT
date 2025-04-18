from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

model = YOLO("app/model/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    label = results[0].names[results[0].probs.top1]  # e.g., "Ripe"
    conf = float(results[0].probs.top1conf) * 100

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": f"{label} ({conf:.2f}% confidence)"}
    )
