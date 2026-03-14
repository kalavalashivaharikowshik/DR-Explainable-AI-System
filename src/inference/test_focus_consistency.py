from src.explainability.focus_consistency import generate_focus_consistency

result = generate_focus_consistency("data/processed/test/2/ee74c3b177e0-GF.jpg")

print("Prediction:", result["prediction"])
print("Consistency score:", result["consistency_score"])
print("Label:", result["consistency_label"])
print("Image saved at:", result["output_path"])