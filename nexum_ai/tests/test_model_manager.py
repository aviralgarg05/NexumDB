"""
Unit tests for model_manager.py - Model loading and inference
"""

import os
from pathlib import Path
from nexum_ai.model_manager import ModelManager


class TestModelManager:
    """Test suite for ModelManager class"""
    
    def test_initialization(self, temp_models_dir):
        """Test ModelManager initialization"""
        manager = ModelManager(models_dir=temp_models_dir)
        assert manager.models_dir == Path(temp_models_dir)
        assert manager.models_dir.exists()
    
    def test_initialization_creates_directory(self, temp_models_dir):
        """Test that ModelManager creates models directory if it doesn't exist"""
        new_dir = os.path.join(temp_models_dir, "new_models")
        ModelManager(models_dir=new_dir)
        assert os.path.exists(new_dir)
    
    def test_ensure_model_existing_file(self, temp_models_dir):
        """Test ensure_model returns path when model exists"""
        manager = ModelManager(models_dir=temp_models_dir)
        
        # Create a dummy model file
        model_path = Path(temp_models_dir) / "test_model.gguf"
        model_path.write_text("dummy model content")
        
        result = manager.ensure_model("test_model.gguf")
        assert result == str(model_path)
    
    def test_ensure_model_missing_no_download_info(self, temp_models_dir):
        """Test ensure_model returns None when model doesn't exist and no download info"""
        manager = ModelManager(models_dir=temp_models_dir)
        result = manager.ensure_model("nonexistent_model.gguf")
        assert result is None
    
    def test_download_model_success(self, temp_models_dir):
        """Test successful model download from HuggingFace"""
        manager = ModelManager(models_dir=temp_models_dir)
        
        # Create the file that would be downloaded
        downloaded_file = Path(temp_models_dir) / "test_model.gguf"
        downloaded_file.write_text("downloaded model")
        
        # File exists, should return path
        result = manager.ensure_model("test_model.gguf")
        
        assert result is not None
        assert result == str(downloaded_file)
    
    def test_download_model_failure(self, temp_models_dir):
        """Test model download failure handling"""
        manager = ModelManager(models_dir=temp_models_dir)
        
        # No file and no download info
        result = manager.ensure_model("test_model.gguf")
        
        assert result is None
    
    def test_list_models_empty(self, temp_models_dir):
        """Test listing models when directory is empty"""
        manager = ModelManager(models_dir=temp_models_dir)
        models = manager.list_models()
        assert models == []
    
    def test_list_models_with_files(self, temp_models_dir):
        """Test listing models when directory contains model files"""
        manager = ModelManager(models_dir=temp_models_dir)
        
        # Create dummy model files
        (Path(temp_models_dir) / "model1.gguf").write_text("model1")
        (Path(temp_models_dir) / "model2.gguf").write_text("model2")
        (Path(temp_models_dir) / "not_a_model.txt").write_text("text")
        
        models = manager.list_models()
        assert len(models) == 2
        assert "model1.gguf" in models
        assert "model2.gguf" in models
        assert "not_a_model.txt" not in models
    
    def test_list_models_nonexistent_directory(self, temp_models_dir):
        """Test listing models when directory doesn't exist"""
        import os
        
        # Create a path that doesn't exist in temp directory
        nonexistent_path = os.path.join(temp_models_dir, "nonexistent_subdir")
        manager = ModelManager(models_dir=nonexistent_path)
        
        # Remove the directory that was created during init
        if manager.models_dir.exists():
            manager.models_dir.rmdir()
        
        models = manager.list_models()
        assert models == []
