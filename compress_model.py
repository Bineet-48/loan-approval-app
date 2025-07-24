import joblib

# Load the original (large) model
model = joblib.load("loan_model.pkl")

# Save a compressed version
joblib.dump(model, "loan_model_compressed.pkl", compress=3)
