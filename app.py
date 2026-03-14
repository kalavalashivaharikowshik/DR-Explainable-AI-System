import gradio as gr
from src.inference.pipeline import run_full_analysis


def analyze_retina(image):

    results = run_full_analysis(image)

    return (
        results["prediction"],
        results["gradcam_heatmap"],
        results["evidence_map"],
        results["consistency_image"],
        results["region_map"],
        results["saliency_map"],
        results["quality_overlay"]
    )


interface = gr.Interface(
    fn=analyze_retina,
    inputs=gr.Image(type="filepath", label="Upload Retina Image"),
    outputs=[
        gr.Text(label="Predicted Severity"),
        gr.Image(label="GradCAM Heatmap"),
        gr.Image(label="Evidence Map"),
        gr.Image(label="Focus Consistency"),
        gr.Image(label="Region Importance"),
        gr.Image(label="Saliency Behavior"),
        gr.Image(label="Quality Overlay")
    ],
    title="Explainable AI System for Diabetic Retinopathy",
    description="Upload a retinal fundus image to analyze diabetic retinopathy severity with explainable AI visualizations."
)

interface.launch()