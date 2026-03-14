from src.explainability.region_importance_map import generate_region_importance

result = generate_region_importance("data/processed/test/2/ee74c3b177e0-GF.jpg")

print("Prediction:", result["prediction"])
print("Macula importance:", result["macula_importance"])
print("Mid retina importance:", result["mid_importance"])
print("Peripheral importance:", result["peripheral_importance"])
print("Saved image:", result["output_path"])