"""FastAPI application for model inference."""
from __future__ import annotations

import io
import os
from pathlib import Path

import torch
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import SegFormerLightningModule


# Initialize logger
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="DL Boilerplate API",
    description="Deep Learning inference API using PyTorch Lightning models",
    version="1.0.0",
)

# Global variables for model and preprocessing
model: Optional[torch.nn.Module] = None
transform: Optional[A.Compose] = None
class_names: Optional[List[str]] = None
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    message: str
    predictions: list[dict[str, any]] | None = None
    processing_time: float | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


def load_model(model_path: str, model_type: str = "segformer") -> torch.nn.Module:
    """Load trained model from checkpoint."""
    try:
        if model_type.lower() == "segformer":
            model = SegFormerLightningModule.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.eval()
        model.to(device)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def setup_preprocessing(image_size: tuple = (256, 256)) -> A.Compose:
    """Setup image preprocessing pipeline for segmentation."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess single image for inference."""
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Apply transformations
    transformed = transform(image=image_array)
    image_tensor = transformed["image"]
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def postprocess_predictions(
    logits: torch.Tensor,
    original_size: tuple,
    apply_softmax: bool = True,
) -> dict[str, any]:
    """Postprocess model predictions."""
    # Apply softmax to get probabilities
    if apply_softmax:
        probabilities = torch.softmax(logits, dim=1)
    else:
        probabilities = logits
    
    # Get class predictions
    predictions = torch.argmax(logits, dim=1)
    
    # Resize to original image size
    if logits.shape[-2:] != original_size:
        import torch.nn.functional as F
        predictions = F.interpolate(
            predictions.unsqueeze(1).float(),
            size=original_size,
            mode="nearest",
        ).squeeze(1).long()
        
        probabilities = F.interpolate(
            probabilities,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
    
    return {
        "predictions": predictions.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "max_probability": probabilities.max().item(),
        "predicted_class": predictions.mode().values.item(),
    }


@app.on_event("startup")
async def startup_event():
    """Initialize model and preprocessing on startup."""
    global model, transform, class_names
    
    # Model configuration (these could come from environment variables)
    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.ckpt")
    model_type = os.getenv("MODEL_TYPE", "segformer")
    image_size = (int(os.getenv("IMAGE_HEIGHT", "256")), int(os.getenv("IMAGE_WIDTH", "256")))
    
    # Load class names
    class_names = ["background", "pet", "border"]
    
    try:
        # Setup preprocessing
        transform = setup_preprocessing(image_size)
        logger.info("Preprocessing pipeline initialized")
        
        # Load model if path exists
        if os.path.exists(model_path):
            model = load_model(model_path, model_type)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model path {model_path} does not exist. API will run without model.")
            
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict on uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Load and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        original_size = image.size[::-1]  # (height, width)
        
        # Preprocess
        image_tensor = preprocess_image(image).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(image_tensor)
        
        # Postprocess
        results = postprocess_predictions(logits, original_size)
        
        processing_time = time.time() - start_time
        
        # Format response
        response_data = {
            "predicted_class": int(results["predicted_class"]),
            "predicted_class_name": class_names[results["predicted_class"]] if class_names else None,
            "confidence": float(results["max_probability"]),
            "image_shape": list(original_size),
            "prediction_shape": list(results["predictions"].shape),
        }
        
        return PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            predictions=[response_data],
            processing_time=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return PredictionResponse(
            success=False,
            message=f"Prediction failed: {str(e)}",
        )


@app.post("/predict_batch", response_model=PredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict on multiple uploaded images."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        import time
        start_time = time.time()
        
        all_predictions = []
        
        for i, file in enumerate(files):
            # Validate file type
            if not file.content_type.startswith("image/"):
                continue
            
            # Load and preprocess image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            original_size = image.size[::-1]  # (height, width)
            
            # Preprocess
            image_tensor = preprocess_image(image).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(image_tensor)
            
            # Postprocess
            results = postprocess_predictions(logits, original_size)
            
            # Format response
            response_data = {
                "image_index": i,
                "filename": file.filename,
                "predicted_class": int(results["predicted_class"]),
                "predicted_class_name": class_names[results["predicted_class"]] if class_names else None,
                "confidence": float(results["max_probability"]),
                "image_shape": list(original_size),
                "prediction_shape": list(results["predictions"].shape),
            }
            
            all_predictions.append(response_data)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            success=True,
            message=f"Batch prediction completed for {len(all_predictions)} images",
            predictions=all_predictions,
            processing_time=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return PredictionResponse(
            success=False,
            message=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "device": str(device),
        "num_classes": len(class_names) if class_names else None,
        "class_names": class_names,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
