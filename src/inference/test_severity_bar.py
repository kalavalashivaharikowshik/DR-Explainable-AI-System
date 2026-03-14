from src.visualization.severity_progress_bar import generate_severity_bar

result = generate_severity_bar("data/processed/test/2/ee74c3b177e0-GF.jpg")

print("Prediction:", result["prediction"])
print("Probabilities:", result["probabilities"])
print("Saved image:", result["output_path"])