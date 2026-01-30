"""
Handwriting Enhancement Module
Preprocesses handwritten text in medical reports for better OCR accuracy
"""

import cv2
import numpy as np


class HandwriteEnhancer:
    def __init__(self):
        """Initialize handwriting enhancement module"""
        pass
    
    def enhance_image(self, image):
        """
        Apply preprocessing to enhance handwritten text
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
    
    def process_file(self, input_path, output_path=None):
        """
        Process an image file
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image (optional)
            
        Returns:
            Enhanced image array
        """
        image = cv2.imread(input_path)
        enhanced = self.enhance_image(image)
        
        if output_path:
            cv2.imwrite(output_path, enhanced)
        
        return enhanced
