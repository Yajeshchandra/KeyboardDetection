from ultralytics import YOLO
import torch
import yaml
import os
from pathlib import Path
import logging
import json
import shutil
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self, data_dir: str, class_mapping_file: str):
        self.data_dir = Path(data_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_class_mapping(class_mapping_file)
        self.setup_directories()
        self.split_data()

    def setup_class_mapping(self, class_mapping_file: str):
        """Load and process class mapping from JSON file"""
        with open(class_mapping_file, 'r') as f:
            data = json.load(f)
        
        # Extract class names and IDs from categories
        self.class_mapping = {
            cat['name']: cat['id']
            for cat in data['categories']
        }
        
        logger.info(f"Loaded {len(self.class_mapping)} classes")
        
    def setup_directories(self):
        """Create necessary directories for training"""
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.test_dir = self.data_dir / 'test'
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            (dir_path / 'images').mkdir(parents=True, exist_ok=True)
            (dir_path / 'labels').mkdir(parents=True, exist_ok=True)
            
    def split_data(self):
        """Split the dataset into train, validation, and test sets"""
        images_path = self.data_dir / 'images'
        labels_path = self.data_dir / 'labels'
        
        # List all image files
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        # Split into 80% train, 10% validation, 10% test
        num_train = int(0.8 * len(image_files))
        num_val = int(0.1 * len(image_files))
        
        train_images = image_files[:num_train]
        val_images = image_files[num_train:num_train+num_val]
        test_images = image_files[num_train+num_val:]

        def copy_files(images, dest_dir):
            for image in images:
                label_file = image.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')  # Adjust based on your label extension
                shutil.copy(images_path / image, dest_dir / 'images' / image)
                shutil.copy(labels_path / label_file, dest_dir / 'labels' / label_file)

        # Copy files to the corresponding directories
        copy_files(train_images, self.train_dir)
        copy_files(val_images, self.val_dir)
        copy_files(test_images, self.test_dir)

    def create_data_yaml(self):
        """Create YAML configuration file for training"""
        data_yaml = {
            'path': str(self.data_dir.absolute()),
            'train': str(self.train_dir / 'images'),
            'val': str(self.val_dir / 'images'),
            'test': str(self.test_dir / 'images'),
            'names': {v: k for k, v in self.class_mapping.items()},
            'nc': len(self.class_mapping)
        }
        
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
            
        logger.info(f"Created data configuration at {yaml_path}")
        return yaml_path
        
    def setup_model(self, model_size: str = 'n'):
        """Initialize YOLO model with specified size"""
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        logger.info(f"Initialized YOLOv8 model: {model_name}")
        
    def train(self, epochs: int = 100, batch_size: int = 16, image_size: int = 640):
        """Train the model with specified parameters"""
        if not hasattr(self, 'model'):
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        # Get GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            logger.info(f"Training on GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        
        # Configure training parameters
        params = {
            'data': str(self.create_data_yaml()),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'device': self.device,
            'workers': min(8, os.cpu_count() or 1),
            'patience': 50,  # Early stopping patience
            'optimizer': 'auto',
            'verbose': True,
            'exist_ok': True,
            'pretrained': True,
            'augment': True,  # Use data augmentation
            'cache': True,  # Cache images for faster training
        }
        
        logger.info("Starting training with parameters:")
        for k, v in params.items():
            logger.info(f"{k}: {v}")
            
        # Start training
        try:
            results = self.model.train(**params)
            logger.info("Training completed successfully")
            return results
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def validate(self):
        """Validate the trained model"""
        if not hasattr(self, 'model'):
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        try:
            results = self.model.val()
            logger.info("Validation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

def main():
    # Initialize trainer
    trainer = YOLOTrainer(
        data_dir='augmented_dataset_yolo',
        class_mapping_file='annotated_dataset/train/_annotations.coco.json'
    )
    
    # Setup and train model
    trainer.setup_model(model_size='n')  # Use nano model for faster training
    results = trainer.train(
        epochs=100,
        batch_size=16,
        image_size=640
    )
    
    # Validate model
    val_results = trainer.validate()
    
    logger.info("Training and validation pipeline completed")

if __name__ == "__main__":
    main()
