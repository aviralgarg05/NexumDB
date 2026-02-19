"""
Model Manager for automatic download and caching of quantized LLMs
"""

import os
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages local model files with automatic downloading from HuggingFace
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelManager
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ModelManager initialized with directory: {self.models_dir}")
    
    def ensure_model(
        self,
        model_name: str,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Ensure model exists locally, download if needed
        
        Args:
            model_name: Local filename for the model
            repo_id: HuggingFace repository ID (e.g., "TheBloke/phi-2-GGUF")
            filename: Filename in the repository (e.g., "phi-2.Q4_K_M.gguf")
        
        Returns:
            Path to local model file, or None if download fails
        """
        local_path = self.models_dir / model_name
        
        if local_path.exists():
            logger.debug(f"Model found at {local_path}")
            return str(local_path)
        
        if repo_id and filename:
            return self._download_model(repo_id, filename, local_path)
        
        logger.warning(f"Model not found and no download info provided: {model_name}")
        return None
    
    def _download_model(self, repo_id: str, filename: str, local_path: Path) -> Optional[str]:
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID
            filename: Filename in the repository
            local_path: Where to save the model
        
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Downloading {filename} from {repo_id}...")
            logger.info("This may take several minutes for large models...")
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False
            )
            
            final_path = self.models_dir / filename
            if final_path.exists():
                logger.debug(f"Model downloaded successfully to {final_path}")
                return str(final_path)
            else:
                logger.warning("Model download completed but file not found at expected location")
                return downloaded_path if os.path.exists(downloaded_path) else None
                
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface-hub")
            return None
        except Exception:
            logger.exception(
                "Error downloading model. "
                "Please check your internet connection and HuggingFace credentials."
            )
            return None
    
    def list_models(self) -> list[str]:
        """List all models in the models directory"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return []
        
        models = list(self.models_dir.glob("*.gguf"))
        return [str(m.name) for m in models]


def test_model_manager():
    """Test the ModelManager with a small model"""
    manager = ModelManager()
    
    logger.info("Testing ModelManager:")
    logger.info("=" * 60)
    
    logger.info("Listing existing models:")
    models = manager.list_models()
    if models:
        for model in models:
            logger.info(f"  - {model}")
    else:
        logger.info("  No models found")
    
    logger.info("Note: Automatic download disabled in test mode")
    logger.info("To download a model, call:")
    logger.info("  manager.ensure_model('phi-2.Q4_K_M.gguf', 'TheBloke/phi-2-GGUF', 'phi-2.Q4_K_M.gguf')")


if __name__ == "__main__":
    test_model_manager()
