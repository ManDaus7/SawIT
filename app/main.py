from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model = YOLO("app/model/best.pt")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    upload_dir = "app/static/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO prediction
    results = model(file_path)
    results.save(save_dir=upload_dir)  # Save annotated image

    # Find the new annotated file
    predicted_image = os.path.join(upload_dir, os.path.basename(results.save_dir[0])) if hasattr(results, 'save_dir') else file.filename

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded_image": f"/static/uploads/{file.filename}",
    })
