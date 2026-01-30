"""
Risk Guard
Rule-based validation layer for medical recommendations
"""

import re


class RiskGuard:
    def __init__(self):
        """Initialize risk guard with validation rules"""
        self.risk_rules = self._load_risk_rules()
        self.contraindications = self._load_contraindications()
    
    def _load_risk_rules(self):
        """
        Load risk detection rules
        
        Returns:
            dict: Risk rules
        """
        # TODO: Load from configuration file
        rules = {
            'high_blood_pressure': {
                'indicators': ['收缩压', '舒张压'],
                'thresholds': {'收缩压': 140, '舒张压': 90},
                'severity': 'high'
            },
            'high_blood_sugar': {
                'indicators': ['血糖', '空腹血糖'],
                'thresholds': {'血糖': 6.1, '空腹血糖': 6.1},
                'severity': 'high'
            },
            'abnormal_liver_function': {
                'indicators': ['ALT', 'AST', '转氨酶'],
                'thresholds': {'ALT': 40, 'AST': 40},
                'severity': 'medium'
            }
        }
        return rules
    
    def _load_contraindications(self):
        """
        Load medical contraindications
        
        Returns:
            dict: Contraindication rules
        """
        # TODO: Load from configuration file
        contraindications = {
            'pregnancy': ['X射线', 'CT', '某些药物'],
            'kidney_disease': ['某些抗生素', '高蛋白饮食'],
            'liver_disease': ['某些止痛药', '酒精']
        }
        return contraindications
    
    def validate_recommendation(self, recommendation, user_profile):
        """
        Validate LLM-generated recommendation against rules
        
        Args:
            recommendation: Generated recommendation text
            user_profile: User profile with medical history
            
        Returns:
            dict: Validation result with warnings
        """
        result = {
            'approved': True,
            'warnings': [],
            'blocked_content': []
        }
        
        # Check for contraindications
        contraindication_warnings = self._check_contraindications(
            recommendation, 
            user_profile
        )
        result['warnings'].extend(contraindication_warnings)
        
        # Check for dangerous recommendations
        dangerous_patterns = [
            r'停止.*治疗',
            r'不需要.*就医',
            r'忽略.*症状',
            r'自行.*手术'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, recommendation):
                result['approved'] = False
                result['blocked_content'].append(f"检测到危险建议: {pattern}")
        
        return result
    
    def _check_contraindications(self, recommendation, user_profile):
        """
        Check for contraindications in recommendation
        
        Args:
            recommendation: Recommendation text
            user_profile: User profile
            
        Returns:
            list: Warning messages
        """
        warnings = []
        
        # Extract user conditions from profile
        conditions = user_profile.get('chronic_conditions', [])
        
        for condition in conditions:
            if condition in self.contraindications:
                contraindicated_items = self.contraindications[condition]
                for item in contraindicated_items:
                    if item in recommendation:
                        warnings.append(
                            f"警告: 患有{condition}的患者应谨慎使用{item}"
                        )
        
        return warnings
    
    def assess_risk_level(self, structured_report):
        """
        Assess overall risk level from report
        
        Args:
            structured_report: Structured medical report
            
        Returns:
            dict: Risk assessment
        """
        assessment = {
            'overall_risk': 'low',
            'identified_risks': [],
            'urgent_attention_needed': []
        }
        
        tests = structured_report.get('tests', [])
        
        for rule_name, rule in self.risk_rules.items():
            for test in tests:
                test_name = test.get('name', '')
                if test_name in rule['indicators']:
                    try:
                        value = float(test.get('value', 0))
                        threshold = rule['thresholds'].get(test_name, float('inf'))
                        
                        if value > threshold:
                            risk_info = {
                                'rule': rule_name,
                                'test': test_name,
                                'value': value,
                                'threshold': threshold,
                                'severity': rule['severity']
                            }
                            assessment['identified_risks'].append(risk_info)
                            
                            if rule['severity'] == 'high':
                                assessment['urgent_attention_needed'].append(risk_info)
                                assessment['overall_risk'] = 'high'
                    except (ValueError, TypeError):
                        continue
        
        return assessment
