from src.explainability.layered_evidence_map import generate_layered_evidence

result = generate_layered_evidence("data/processed/test/2/ee74c3b177e0-GF.jpg")

print("Prediction:", result["prediction"])
print("Evidence map saved at:", result["evidence_map"])