"""
LLM Inference Module
Handles inference with fine-tuned Qwen3-8B model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class MedicalLLMInference:
    def __init__(self, model_path, device='auto'):
        """
        Initialize LLM inference
        
        Args:
            model_path: Path to fine-tuned model
            device: Device to run inference on
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16
        )
        self.model.eval()
    
    def generate_report_analysis(self, structured_report, user_profile=None, context=None):
        """
        Generate medical report analysis
        
        Args:
            structured_report: Structured medical report data
            user_profile: User health profile (optional)
            context: Retrieved context from RAG (optional)
            
        Returns:
            str: Generated analysis
        """
        prompt = self._build_prompt(structured_report, user_profile, context)
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def _build_prompt(self, structured_report, user_profile, context):
        """
        Build prompt for LLM
        
        Args:
            structured_report: Structured report data
            user_profile: User profile
            context: Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        prompt_parts = ["你是一个专业的医疗分析助手。请分析以下体检报告并提供专业建议。\n"]
        
        # Add context if available
        if context and context.get('knowledge'):
            prompt_parts.append("参考医学知识:\n")
            for doc in context['knowledge'][:3]:
                prompt_parts.append(f"- {doc['text']}\n")
            prompt_parts.append("\n")
        
        # Add user profile if available
        if user_profile:
            prompt_parts.append("用户健康档案:\n")
            if user_profile.get('chronic_conditions'):
                prompt_parts.append(f"慢性疾病: {', '.join(user_profile['chronic_conditions'])}\n")
            if user_profile.get('abnormal_indicators'):
                prompt_parts.append("历史异常指标: ")
                prompt_parts.append(str(user_profile['abnormal_indicators'][:3]))
                prompt_parts.append("\n")
            prompt_parts.append("\n")
        
        # Add current report
        prompt_parts.append("当前体检报告:\n")
        if structured_report.get('tests'):
            for test in structured_report['tests']:
                test_line = f"- {test.get('name', 'N/A')}: {test.get('value', 'N/A')} {test.get('unit', '')}"
                if 'reference_range' in test:
                    test_line += f" (参考范围: {test['reference_range']})"
                prompt_parts.append(test_line + "\n")
        
        prompt_parts.append("\n请提供:\n1. 异常指标分析\n2. 健康风险评估\n3. 具体建议\n\n回答:")
        
        return ''.join(prompt_parts)
    
    def generate_qa_response(self, question, context):
        """
        Generate answer to user question with RAG context
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            str: Generated answer
        """
        prompt = f"""基于以下医学知识和用户健康档案,回答用户问题。

医学知识:
{self._format_context(context.get('knowledge', []))}

用户档案:
{self._format_context(context.get('profile', []))}

用户问题: {question}

请提供专业、准确的回答:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def _format_context(self, context_list):
        """Format context list for prompt"""
        if not context_list:
            return "无"
        
        formatted = []
        for i, item in enumerate(context_list[:5], 1):
            formatted.append(f"{i}. {item.get('text', '')}")
        
        return '\n'.join(formatted)
