import sys
import logging 
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np 
from PIL import Image 
import torch

from paddleocr import PaddleOCR

from config import config

logging.basicConfig(level = logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OCREngine")

def resolve_device(request_device: str) -> bool:
    if request_device == "mps":
        try: 
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("Apple Silicon MPS acceleration detected and enabled.")
                return True
            else:
                logger.warning("MPS requested but not available on this system. Falling back to CPU.")
                return False
        except ImportError:
            logger.warning("PyTorch not installed to verify MPS state. Defaulting device check to Paddle settings.")
            return True

class OCREngine:
    def __init__(self):
        self.use_gpu = resolve_device(config.device)
        
        self.ocr = PaddleOCR(
            use_angle_cls = config.use_angle_cls,
            lang = config.lang,
            use_gpu = self.use_gpu,
            show_log = False
        )
        
    def extract_text(self, image_input:Union[str, Path, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        if isinstance(image_input, Path):
            image_input = str(image_input)
        elif isinstance(image_input, Image.Image):
            image_input = np.array(image_input)
        
        results = self.ocr.ocr(image_input, cls = config.use_angle_cls)
        
        parsed_output = []
        if not results or results[0] is None:
            logger.warning("No text found in image")
            return parsed_output 
        
        for line in results[0]:
            bbox, (text, confidence) = line
            parsed_output.append({
                "text" : text,
                "confidence" : float(confidence),
                "bbox" : bbox
            })
        return parsed_output