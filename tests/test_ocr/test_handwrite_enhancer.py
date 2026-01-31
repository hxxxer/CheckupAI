import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from backend.ocr.handwrite_enhancer import HandwriteEnhancer

class TestHandwriteEnhancer(unittest.TestCase):
    def setUp(self):
        self.enhancer = HandwriteEnhancer()

    def test_enhance_image_bgr(self):
        # Create a dummy BGR image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        enhanced = self.enhancer.enhance_image(dummy_image)
        
        # Check if output is a numpy array and has correct shape (binary image is 2D)
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertEqual(enhanced.shape, (100, 100))

    def test_enhance_image_gray(self):
        # Create a dummy grayscale image
        dummy_image = np.zeros((100, 100), dtype=np.uint8)
        enhanced = self.enhancer.enhance_image(dummy_image)
        
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertEqual(enhanced.shape, (100, 100))

    @patch('cv2.imread')
    @patch('cv2.imwrite')
    def test_process_file(self, mock_imwrite, mock_imread):
        # Mock imread to return a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = dummy_image
        
        input_path = "dummy_input.jpg"
        output_path = "dummy_output.jpg"
        
        enhanced = self.enhancer.process_file(input_path, output_path)
        
        mock_imread.assert_called_once_with(input_path)
        mock_imwrite.assert_called_once()
        self.assertIsInstance(enhanced, np.ndarray)

if __name__ == '__main__':
    unittest.main()
