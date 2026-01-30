"""
LLM Validator
Uses LLM to validate and complete NER results
"""

from openai import OpenAI
import json


class LLMValidator:
    def __init__(self, api_key=None, base_url=None, model='gpt-3.5-turbo'):
        """
        Initialize LLM validator
        
        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def validate_and_complete(self, structured_data, raw_text):
        """
        Validate and complete structured data using LLM
        
        Args:
            structured_data: Structured data from NER
            raw_text: Original text for reference
            
        Returns:
            dict: Validated and completed structured data
        """
        prompt = f"""
你是一个医疗数据验证助手。请验证并完善以下从体检报告中提取的结构化数据。

原始文本:
{raw_text}

提取的结构化数据:
{json.dumps(structured_data, ensure_ascii=False, indent=2)}

请执行以下任务:
1. 验证提取的数据是否准确
2. 补充遗漏的重要信息
3. 修正明显的错误
4. 标准化单位和格式

返回格式: JSON对象,包含validated(布尔值)和corrected_data(完善后的结构化数据)
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗数据分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            print(f"LLM validation error: {e}")
            return {
                'validated': False,
                'corrected_data': structured_data,
                'error': str(e)
            }
    
    def extract_missing_fields(self, structured_data, raw_text, required_fields):
        """
        Extract missing required fields using LLM
        
        Args:
            structured_data: Current structured data
            raw_text: Original text
            required_fields: List of required field names
            
        Returns:
            dict: Additional extracted fields
        """
        missing = [f for f in required_fields if f not in structured_data]
        
        if not missing:
            return {}
        
        prompt = f"""
从以下体检报告文本中提取这些缺失的字段: {', '.join(missing)}

文本:
{raw_text}

请返回JSON格式的提取结果。
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个医疗信息提取专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            extracted = json.loads(response.choices[0].message.content)
            return extracted
        
        except Exception as e:
            print(f"Field extraction error: {e}")
            return {}
