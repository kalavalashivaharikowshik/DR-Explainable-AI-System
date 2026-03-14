from src.explainability.gradcam import run_gradcam

result = run_gradcam("data/processed/test/4/1fd5d860d4d7-GF.jpg")

print("Prediction:", result["prediction"])
print("Heatmap saved at:", result["heatmap_path"])