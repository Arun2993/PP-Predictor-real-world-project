from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Load required model files
try:
    # Load TF-IDF Vectorizer
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    # Load StandardScaler (trained on final feature set)
    scaler = joblib.load("scaler.pkl")
    
    # Load Logistic Regression Model
    model = joblib.load("logistic_regression_model.pkl")

    print("✅ Model files loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model files: {e}")
    raise RuntimeError("Failed to load required model files.")

# Define input data model
class InputData(BaseModel):
    Pantone_Colors_Count: int
    Total_Colors_Found: int
    White_Count: int
    Varnish_Count: int
    Digital_or_Other: int
    Instructions_Extracted: str
    Eligible_to_Bypass_Auto_Trap: int
    Color_Eligible_to_Bypass: int

# API endpoint for predictions
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Extract numerical features
        numeric_features = np.array([[
            data.Pantone_Colors_Count,
            data.Total_Colors_Found,
            data.White_Count,
            data.Varnish_Count,
            data.Digital_or_Other,
            data.Eligible_to_Bypass_Auto_Trap,
            data.Color_Eligible_to_Bypass
        ]])

        # Transform text feature using TF-IDF
        text_feature = tfidf_vectorizer.transform([data.Instructions_Extracted]).toarray()

        # Debugging: Print feature shapes
        print(f"Numeric Features Shape: {numeric_features.shape}")  # (1, 7)
        print(f"Text Feature Shape: {text_feature.shape}")  # (1, N) where N is TF-IDF vocab size

        # Ensure the final shape matches training shape (557)
        expected_features = scaler.mean_.shape[0]  # 557 features expected
        actual_features = numeric_features.shape[1] + text_feature.shape[1]

        if actual_features < expected_features:
            # Pad with zeros if features are missing
            padding = np.zeros((1, expected_features - actual_features))
            final_features = np.hstack((numeric_features, text_feature, padding))
            print(f"⚠️ Warning: Added {expected_features - actual_features} zero-padding features.")
        elif actual_features > expected_features:
            # Trim extra features if more than expected
            final_features = np.hstack((numeric_features, text_feature))[:, :expected_features]
            print(f"⚠️ Warning: Trimmed {actual_features - expected_features} extra features.")
        else:
            final_features = np.hstack((numeric_features, text_feature))

        # Debugging: Print final shape before scaling
        print(f"Final Feature Shape Before Scaling: {final_features.shape}")  # Should match 557

        # Scale the combined features
        final_features_scaled = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features_scaled)[0]

        return {"Eligible_to_Bypass_PP": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run FastAPI with: uvicorn main:app --reload
