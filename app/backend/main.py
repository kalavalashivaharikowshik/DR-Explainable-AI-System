from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os

from src.inference.pipeline import run_full_analysis
from src.email_service.send_email import send_email

app = FastAPI()

# Ensure required directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Static folders
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Upload folder
UPLOAD_DIR = "app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "DR Explainable AI System API running"}


@app.post("/analyze")
async def analyze_retina(
    file: UploadFile = File(...),
    email: str = Form(None)
):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run AI pipeline
    results = run_full_analysis(file_path)

    # Send email if provided
    if email:
        send_email(
            email,
            results["report_pdf"],
            results["pdf_password"]
        )

    return results


@app.get("/reference/{folder_id}")
def get_reference_images(folder_id: int):

    folder = f"data/reference_images/{folder_id}"

    images = []

    if os.path.exists(folder):
        for file in os.listdir(folder):
            images.append(f"/data/reference_images/{folder_id}/{file}")

    return {"images": images}


@app.post("/send-report")
def send_report(data: dict = Body(...)):

    email = data["email"]
    pdf_path = data["pdf_path"]
    password = data["password"]

    send_email(email, pdf_path, password)

    return {"status": "email_sent"}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=port)
