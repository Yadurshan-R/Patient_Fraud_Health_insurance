"""
FastAPI Service for ML-Powered Insurance Fraud Detection
Simplified version without database - Standalone ML API
"""

from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import logging
import tempfile
import os

from model_loader import get_model, FraudDetectionModel
from prescription_verifier import verify_prescription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HealthTrust ML Service",
    description="ML-powered fraud detection for insurance claims",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    model = get_model()  # Warm up model
    logger.info(f"✓ ML Service started with {model.metadata['model_name']}")


# Pydantic models
class ClaimRequest(BaseModel):
    """Request model for claim prediction"""
    amount_billed: float = Field(..., description="Claim amount", gt=0)
    age: int = Field(..., description="Patient age", gt=0, lt=120)
    gender: str = Field(..., description="Patient gender (Male/Female)")
    diagnosis: str = Field(..., description="Diagnosis category")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('diagnosis')
    def validate_diagnosis(cls, v):
        valid_diagnoses = [
            'Pregnancy', 'Hypertension', 'Diabetes', 'Pneumonia',
            'Gastroenteritis', 'Cesarean Section', 'Cataract Surgery', 'Other'
        ]
        if v not in valid_diagnoses:
            raise ValueError(f'Diagnosis must be one of: {", ".join(valid_diagnoses)}')
        return v


class ClaimResponse(BaseModel):
    """Response model for claim prediction"""
    prediction: int  # 0 = Genuine, 1 = Fraud
    prediction_label: str  # 'genuine' or 'fake'
    confidence: float
    message: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    accuracy: float
    f1_score: float
    timestamp: str


# Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check(model: FraudDetectionModel = Depends(get_model)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=model.metadata['model_name'],
        accuracy=model.metadata['accuracy'],
        f1_score=model.metadata['f1_score'],
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=ClaimResponse, status_code=status.HTTP_200_OK)
async def predict_claim(
    claim: ClaimRequest,
    model: FraudDetectionModel = Depends(get_model)
):
    """
    Submit a claim and get ML fraud prediction
    """
    try:
        # Get ML prediction
        prediction_result = model.predict(
            amount_billed=claim.amount_billed,
            age=claim.age,
            gender=claim.gender,
            diagnosis=claim.diagnosis
        )
        
        # Prepare response message
        if prediction_result['prediction_label'] == 'fake':
            message = f"Claim REJECTED - Detected as fraudulent with {prediction_result['confidence']:.1%} confidence"
        else:
            message = f"Claim APPROVED - Verified as genuine with {prediction_result['confidence']:.1%} confidence"
        
        # Terminal output
        print("\n" + "="*80)
        print("🏥 INSURANCE CLAIM PREDICTION")
        print("="*80)
        print(f"\n📋 CLAIM DETAILS:")
        print(f"   Amount Billed:      ₳{claim.amount_billed:,.2f}")
        print(f"   Patient Age:        {claim.age} years")
        print(f"   Gender:             {claim.gender}")
        print(f"   Diagnosis:          {claim.diagnosis}")
        print(f"\n🤖 AI FRAUD DETECTION:")
        print(f"   Prediction:         {prediction_result['prediction_label'].upper()}")
        print(f"   Confidence:         {prediction_result['confidence']:.2%}")
        
        if prediction_result['prediction_label'] == 'genuine':
            print(f"\n✅ VERDICT: CLAIM APPROVED")
        else:
            print(f"\n❌ VERDICT: CLAIM REJECTED")
        
        print("="*80 + "\n")
        
        logger.info(f"Prediction: {prediction_result['prediction_label']} ({prediction_result['confidence']:.2%})")
        
        return ClaimResponse(
            prediction=prediction_result['prediction'],
            prediction_label=prediction_result['prediction_label'],
            confidence=prediction_result['confidence'],
            message=message,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/verify-images")
async def verify_images(
    prescription_image: UploadFile = File(...),
    receipt_image: UploadFile = File(...),
    ml_confidence: Optional[float] = Form(None)
):
    """
    Verify prescription and receipt images
    Returns combined score if ml_confidence is provided
    """
    try:
        print("\n" + "="*80)
        print(f"📸 IMAGE VERIFICATION")
        print("="*80)
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as prescription_temp:
            prescription_temp.write(await prescription_image.read())
            prescription_path = prescription_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as receipt_temp:
            receipt_temp.write(await receipt_image.read())
            receipt_path = receipt_temp.name
        
        # Run image verification using GPT-4o Vision API
        is_genuine, match_percentage = verify_prescription(prescription_path, receipt_path)
        
        # Clean up temp files
        os.unlink(prescription_path)
        os.unlink(receipt_path)
        
        # Store results
        image_status = "genuine" if is_genuine else "fake"
        image_score = match_percentage
        
        # Calculate combined score if ML confidence provided
        combined_score = None
        final_status = image_status
        
        if ml_confidence is not None:
            ml_confidence_percent = float(ml_confidence) * 100
            combined_score = (ml_confidence_percent + image_score) / 2
            final_status = "genuine" if combined_score >= 80 else "fake"
            
            print(f"\n🎯 COMBINED SCORE:")
            print(f"   ML Confidence:      {ml_confidence_percent:.2f}%")
            print(f"   Image Score:        {image_score:.2f}%")
            print(f"   Combined Average:   {combined_score:.2f}%")
            
            if combined_score >= 80:
                print(f"\n✅ FINAL VERDICT: APPROVED (>= 80%)")
            else:
                print(f"\n❌ FINAL VERDICT: REJECTED (< 80%)")
        else:
            print(f"\n📊 IMAGE RESULTS:")
            print(f"   Match Score:        {image_score:.2f}%")
            print(f"   Status:             {image_status.upper()}")
        
        print("="*80 + "\n")
        
        logger.info(f"Image verification: {image_status} ({image_score:.2f}%)")
        
        response_data = {
            "image_verification": {
                "status": image_status,
                "score": image_score,
                "is_genuine": is_genuine
            },
            "final_status": final_status
        }
        
        if combined_score is not None:
            response_data["combined_score"] = combined_score
            response_data["ml_score"] = ml_confidence_percent
            response_data["message"] = f"Combined score: {combined_score:.1f}% ({'✅ APPROVED' if combined_score >= 80 else '❌ REJECTED'})"
        else:
            response_data["message"] = f"Image verification: {image_score:.1f}% match"
        
        return response_data
        
    except Exception as e:
        logger.error(f"Image verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Image verification failed: {str(e)}")


@app.get("/model/info")
async def get_model_info(model: FraudDetectionModel = Depends(get_model)):
    """Get detailed model information"""
    return {
        "model_name": model.metadata['model_name'],
        "accuracy": model.metadata['accuracy'],
        "f1_score": model.metadata['f1_score'],
        "n_features": model.metadata['n_features'],
        "training_date": model.metadata['training_date'],
        "top_features": model.get_feature_importance(top_n=10)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
