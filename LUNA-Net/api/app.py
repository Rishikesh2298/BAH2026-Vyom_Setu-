import os
import shutil
import tempfile
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from data_pipeline.pds4_parser import parse_pds4_data
from data_pipeline.preprocessor import LunarDataPreprocessor
from models.fusion_network import LUNANetFusion

# Initialize FastAPI App
app = FastAPI(
    title="LUNA-Net Inference API",
    description="Automated Hazard Mapping & Polarimetric Characterization System for Chandrayaan-2 Data.",
    version="1.0.0"
)

# Initialize Model & Preprocessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LUNANetFusion(optical_channels=3, radar_channels=1, embed_dim=512).to(device)
model.eval()  # Set model to evaluation mode

preprocessor = LunarDataPreprocessor(patch_size=256, stride=256)

class PredictionResponse(BaseModel):
    safe_landing_zones: List[Dict[str, int]]
    water_ice_probability: float
    roughness_index: float

@app.get("/health")
async def health_check():
    """
    Returns system performance and model status.
    """
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    }

def _extract_bounding_boxes(mask_array: np.ndarray) -> List[Dict[str, int]]:
    """
    Extracts bounding boxes from a binary segmentation mask.
    The mask_array is expected to be shape (H, W).
    """
    # Threshold just in case, though it's typically binary from the model post-processing
    binary_mask = (mask_array > 0.5).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise boxes
        if w > 5 and h > 5:
            boxes.append({
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h
            })
    return boxes

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    optical_file: UploadFile = File(..., description="PDS4 XML/IMG file for OHRC optical data"),
    radar_file: UploadFile = File(..., description="PDS4 XML/IMG file for DFSAR radar data")
):
    """
    Predicts hazard zones, water ice probability, and surface roughness from input optical and radar files.
    """
    # Create temporary directory to save the uploaded files
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            opt_path = os.path.join(tmpdir, optical_file.filename)
            with open(opt_path, "wb") as buffer:
                shutil.copyfileobj(optical_file.file, buffer)
                
            rad_path = os.path.join(tmpdir, radar_file.filename)
            with open(rad_path, "wb") as buffer:
                shutil.copyfileobj(radar_file.file, buffer)
                
            # 1. Parse PDS4
            opt_data = parse_pds4_data(opt_path)
            rad_data = parse_pds4_data(rad_path)
            
            # 2. Preprocess & Patchify
            # In a real scenario, you'd process all patches. Here we take the first patch for demonstration.
            patches = preprocessor.process_and_patchify(opt_data, rad_data)
            
            if len(patches) == 0:
                raise HTTPException(status_code=400, detail="Generated 0 patches. Input arrays might be too small.")
                
            # For inference endpoint, we process the first valid 256x256 patch
            opt_patch, rad_patch = patches[0]
            
            # Convert to PyTorch tensors and add batch dimension
            # Model expects (B, C, H, W) for ViT and (B, Seq_Len, C, H, W) for LSTM
            
            # (H, W, C) -> (C, H, W)
            opt_tensor = torch.from_numpy(opt_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
            rad_tensor = torch.from_numpy(rad_patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # Mocking sequence length for radar (Seq_Len = 1)
            # rad_tensor is (B, C, H, W). We need (B, Seq_Len, C, H, W)
            rad_seq_tensor = rad_tensor.unsqueeze(1)
            
            # 3. Model Inference
            with torch.no_grad():
                seg_mask, water_prob, roughness = model(opt_tensor, rad_seq_tensor)
                
            # Convert outputs to CPU Numpy for post-processing
            # seg_mask: (1, 1, 256, 256)
            mask_np = seg_mask.squeeze().cpu().numpy()
            water_prob_val = float(water_prob.squeeze().cpu().numpy())
            roughness_val = float(roughness.squeeze().cpu().numpy())
            
            # 4. Extract Bounding Boxes for Safe Landing Zones
            boxes = _extract_bounding_boxes(mask_np)
            
            return PredictionResponse(
                safe_landing_zones=boxes,
                water_ice_probability=water_prob_val,
                roughness_index=roughness_val
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
