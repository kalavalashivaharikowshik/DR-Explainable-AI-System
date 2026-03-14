from src.explainability.saliency_evolution import generate_saliency_evolution

result = generate_saliency_evolution("data/processed/test/2/ee74c3b177e0-GF.jpg")

print("Prediction:", result["prediction"])
print("Saved image:", result["output_path"])