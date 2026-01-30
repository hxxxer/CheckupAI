"""
Medical NER Module
BERT-based Named Entity Recognition for medical reports
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class MedicalNER:
    def __init__(self, model_name='bert-base-chinese'):
        """
        Initialize Medical NER model
        
        Args:
            model_name: Pretrained model name or path
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Medical entity labels
        self.labels = [
            'O',           # Outside
            'B-TEST',      # Beginning of test name
            'I-TEST',      # Inside test name
            'B-VALUE',     # Beginning of value
            'I-VALUE',     # Inside value
            'B-UNIT',      # Beginning of unit
            'I-UNIT',      # Inside unit
            'B-RANGE',     # Beginning of reference range
            'I-RANGE',     # Inside reference range
            'B-SYMPTOM',   # Beginning of symptom
            'I-SYMPTOM',   # Inside symptom
        ]
    
    def extract_entities(self, text):
        """
        Extract medical entities from text
        
        Args:
            text: Input text from medical report
            
        Returns:
            list: Extracted entities with labels
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Decode predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        pred_labels = [self.labels[p] if p < len(self.labels) else 'O' 
                      for p in predictions[0].cpu().numpy()]
        
        # Extract entities
        entities = []
        current_entity = None
        current_text = []
        
        for token, label in zip(tokens, pred_labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append({
                        'type': current_entity,
                        'text': ''.join(current_text).replace('##', '')
                    })
                current_entity = label[2:]
                current_text = [token]
            elif label.startswith('I-') and current_entity:
                current_text.append(token)
            else:
                if current_entity:
                    entities.append({
                        'type': current_entity,
                        'text': ''.join(current_text).replace('##', '')
                    })
                current_entity = None
                current_text = []
        
        if current_entity:
            entities.append({
                'type': current_entity,
                'text': ''.join(current_text).replace('##', '')
            })
        
        return entities
    
    def structure_report(self, text):
        """
        Structure medical report into key-value pairs
        
        Args:
            text: Raw text from medical report
            
        Returns:
            dict: Structured medical data
        """
        entities = self.extract_entities(text)
        
        # Group entities into structured data
        structured = {
            'tests': [],
            'symptoms': []
        }
        
        current_test = {}
        for entity in entities:
            if entity['type'] == 'TEST':
                if current_test:
                    structured['tests'].append(current_test)
                current_test = {'name': entity['text']}
            elif entity['type'] == 'VALUE' and current_test:
                current_test['value'] = entity['text']
            elif entity['type'] == 'UNIT' and current_test:
                current_test['unit'] = entity['text']
            elif entity['type'] == 'RANGE' and current_test:
                current_test['reference_range'] = entity['text']
            elif entity['type'] == 'SYMPTOM':
                structured['symptoms'].append(entity['text'])
        
        if current_test:
            structured['tests'].append(current_test)
        
        return structured
