"""
Paddle Table Parser
Uses PaddleOCR for table structure recognition and text extraction
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np


class PaddleTableParser:
    def __init__(self, use_gpu=False):
        """
        Initialize PaddleOCR with table recognition
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=use_gpu,
            show_log=False
        )
    
    def parse_table(self, image_path):
        """
        Parse table structure and extract text
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Parsed table data with structure and content
        """
        result = self.ocr.ocr(image_path, cls=True)
        
        # Process OCR results
        parsed_data = {
            'text_boxes': [],
            'content': []
        }
        
        if result and len(result) > 0:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                parsed_data['text_boxes'].append({
                    'bbox': box,
                    'text': text,
                    'confidence': confidence
                })
                parsed_data['content'].append(text)
        
        return parsed_data
    
    def extract_structured_data(self, image_path):
        """
        Extract structured data from medical report table
        
        Args:
            image_path: Path to the medical report image
            
        Returns:
            dict: Structured medical report data
        """
        parsed = self.parse_table(image_path)
        
        # TODO: Implement table structure recognition logic
        # This should identify rows, columns, and key-value pairs
        
        return {
            'raw_text': parsed['content'],
            'structured_fields': {}
        }
