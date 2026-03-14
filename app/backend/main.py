from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os

from src.inference.pipeline import run_full_analysis
from fastapi.staticfiles import StaticFiles
from src.email_service.send_email import send_email

app = FastAPI()

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/data", StaticFiles(directory="data"), name="data")

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
    email: str = Form(None),
    password: str = Form(None)
):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run AI pipeline
    results = run_full_analysis(file_path)

    # Send email if provided
    if email:
        send_email(email, results["report_pdf"], password)

    return results

@app.get("/reference/{folder_id}")
def get_reference_images(folder_id: int):

    folder = f"data/reference_images/{folder_id}"

    images = []

    for file in os.listdir(folder):

        images.append(f"/data/reference_images/{folder_id}/{file}")

    return {"images": images}

from fastapi import Body
from src.email_service.send_email import send_email

@app.post("/send-report")

def send_report(data: dict = Body(...)):

    email = data["email"]

    pdf_path = "outputs/report/dr_report_secure.pdf"

    password = "sent_in_email"

    send_email(email, pdf_path, password)

    return {"status":"email_sent"}