import os 
from pathlib import Path 
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen= True)
class Config:
    """Read-only configuration settings loaded from environment variables."""
    
    device : str = os.getenv("DEVICE", "cpu")
    
    input_dir: str = Path(os.getenv("INPUT_DIR"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR"))
    
    def ensure_directories(self) -> None:
        """Create configured directories if they do not exist."""
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        


config = Config()
config.ensure_directories()