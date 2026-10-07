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
