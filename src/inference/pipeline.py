from src.models.classification.efficientnet_model import get_model
import torch
from torchvision import transforms
from PIL import Image
import os

from src.explainability.gradcam import run_gradcam
from src.explainability.layered_evidence_map import generate_layered_evidence
from src.explainability.focus_consistency import generate_focus_consistency
from src.explainability.region_importance_map import generate_region_importance
from src.explainability.saliency_evolution import generate_saliency_evolution
from src.visualization.comparison_panel import get_reference_images
from src.models.classification.efficientnet_model import get_model
from src.visualization.pseudo3d_abstraction import generate_pseudo3d
from src.reporting.generate_pdf import generate_pdf_report
from src.reporting.encrypt_pdf import encrypt_pdf
from src.email_service.send_email import send_email
from src.quality_assessment.retinal_quality_overlay import generate_quality_overlay

def run_full_analysis(image_path):

    results = {}

    # ---------------------------
    # GradCAM
    # ---------------------------
    gradcam_result = run_gradcam(image_path)

    results["prediction"] = gradcam_result["prediction"]
    results["gradcam_heatmap"] = gradcam_result["heatmap_path"]

    # ---------------------------
    # Layered Evidence Map
    # ---------------------------
    evidence_result = generate_layered_evidence(image_path)

    results["evidence_map"] = evidence_result["evidence_map"]

    # ---------------------------
    # Focus Consistency
    # ---------------------------
    consistency_result = generate_focus_consistency(image_path)

    results["consistency_score"] = consistency_result["consistency_score"]
    results["consistency_label"] = consistency_result["consistency_label"]
    results["consistency_image"] = consistency_result["output_path"]

    # ---------------------------
    # Region Importance
    # ---------------------------
    region_result = generate_region_importance(image_path)

    results["macula_importance"] = region_result["macula_importance"]
    results["mid_importance"] = region_result["mid_importance"]
    results["peripheral_importance"] = region_result["peripheral_importance"]
    results["region_map"] = region_result["output_path"]

    # ---------------------------
    # Saliency Evolution
    # ---------------------------
    saliency_result = generate_saliency_evolution(image_path)

    results["saliency_map"] = saliency_result["output_path"]

    references = get_reference_images(results["prediction"])

    results["reference_same"] = references["same_class_example"]
    results["reference_severe"] = references["severe_example"]

    prediction, probs = get_prediction_with_probs(image_path)

    results["prediction"] = prediction
    results["probabilities"] = probs

    pseudo3d = generate_pseudo3d(image_path)

    results["pseudo3d_map"] = pseudo3d["pseudo3d_map"]

    report_path = generate_pdf_report(image_path, results)

    encrypted_pdf, password = encrypt_pdf(report_path)

    results["report_pdf"] = encrypted_pdf
    results["pdf_password"] = password

    quality = generate_quality_overlay(image_path)

    results["quality_overlay"] = quality["quality_map"]

    return results

def get_prediction_with_probs(image_path):

    model = get_model()
    model.load_state_dict(torch.load("models/trained/best_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output,dim=1)[0].tolist()

    classes = ["No DR","Mild","Moderate","Severe","Proliferative"]

    prediction = classes[probs.index(max(probs))]

    return prediction, probs