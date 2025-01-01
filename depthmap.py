import numpy as np
import cv2
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras.models import Model

from keras.applications import ResNet50

def create_depth_model(input_shape=(384, 384, 3)):
    """Creates a basic depth estimation model using DenseNet169 as backbone"""
    # Base model - DenseNet169 without top layers
    base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Create encoder by taking outputs from different layers
    encoder_output = base_model.output
    
    # Decoder
    x = Conv2D(512, 3, padding='same')(encoder_output)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x)
    
    x = Conv2D(256, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x)
    
    x = Conv2D(128, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x)
    
    x = Conv2D(64, 3, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = UpSampling2D()(x)
    
    # Final output - single channel depth map
    outputs = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

def process_image(image_path, model):
    """Process a single image to create its depth map"""
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    
    # Normalize image
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict depth
    depth_map = model.predict(img_batch)[0, :, :, 0]
    
    return depth_map

def save_depth_map(depth_map, output_path):
    """Save depth map as a grayscale image"""
    # Normalize to 0-255 range
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Save image
    cv2.imwrite(output_path, depth_map)

# Example usage
def main():
    # Create model
    model = create_depth_model()
    
    # Process image
    input_image_path = "26.jpg"
    output_path = "depth_map.jpg"
    
    depth_map = process_image(input_image_path, model)
    save_depth_map(depth_map, output_path)