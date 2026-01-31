import unittest
from unittest.mock import patch, MagicMock
from backend.ocr.paddle_table_parser import PaddleTableParser

class TestPaddleTableParser(unittest.TestCase):
    @patch('backend.ocr.paddle_table_parser.PaddleOCR')
    def setUp(self, mock_paddle_ocr):
        self.mock_ocr_instance = MagicMock()
        mock_paddle_ocr.return_value = self.mock_ocr_instance
        self.parser = PaddleTableParser(use_gpu=False)

    def test_parse_table(self):
        # Mock OCR result
        # result[0] = [[box, [text, confidence]], ...]
        mock_result = [[
            [[0, 0], [10, 0], [10, 10], [0, 10]], ['Test Text', 0.95]
        ]]
        self.mock_ocr_instance.ocr.return_value = mock_result
        
        image_path = "dummy_image.jpg"
        parsed_data = self.parser.parse_table(image_path)
        
        self.mock_ocr_instance.ocr.assert_called_once_with(image_path, cls=True)
        self.assertIn('text_boxes', parsed_data)
        self.assertIn('content', parsed_data)
        self.assertEqual(len(parsed_data['content']), 1)
        self.assertEqual(parsed_data['content'][0], 'Test Text')
        self.assertEqual(parsed_data['text_boxes'][0]['text'], 'Test Text')
        self.assertEqual(parsed_data['text_boxes'][0]['confidence'], 0.95)

    def test_extract_structured_data(self):
        # Mock parse_table
        mock_parsed = {
            'text_boxes': [],
            'content': ['Field1', 'Value1']
        }
        with patch.object(PaddleTableParser, 'parse_table', return_value=mock_parsed):
            result = self.parser.extract_structured_data("dummy.jpg")
            
            self.assertEqual(result['raw_text'], ['Field1', 'Value1'])
            self.assertIsInstance(result['structured_fields'], dict)

if __name__ == '__main__':
    unittest.main()
