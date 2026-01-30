"""
Chainlit Application for CheckupAI
Frontend interface for medical report analysis
"""

import chainlit as cl
import aiohttp
import os
from typing import Optional


# API endpoint
API_URL = os.getenv('API_URL', 'http://localhost:8000')


@cl.on_chat_start
async def start():
    """Initialize chat session"""
    await cl.Message(
        content="欢迎使用CheckupAI体检报告分析系统!\n\n请上传您的体检报告图片,我将为您提供专业的分析和建议。"
    ).send()
    
    # Initialize user session
    cl.user_session.set("user_id", None)
    cl.user_session.set("current_report", None)


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    
    # Check if message contains files
    if message.elements:
        await handle_file_upload(message)
    else:
        await handle_question(message)


async def handle_file_upload(message: cl.Message):
    """Handle medical report file upload"""
    
    # Show processing message
    processing_msg = cl.Message(content="正在处理您的体检报告,请稍候...")
    await processing_msg.send()
    
    # Get uploaded file
    file = message.elements[0]
    
    try:
        # Upload file to API
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(file.path, 'rb'),
                          filename=file.name,
                          content_type='image/jpeg')
            
            user_id = cl.user_session.get("user_id")
            if user_id:
                data.add_field('user_id', user_id)
            
            async with session.post(f"{API_URL}/api/upload-report", data=data) as resp:
                upload_result = await resp.json()
        
        # Analyze report
        async with aiohttp.ClientSession() as session:
            payload = {
                "image_path": upload_result['file_path'],
                "user_id": user_id
            }
            
            async with session.post(f"{API_URL}/api/analyze-report", json=payload) as resp:
                if resp.status == 200:
                    analysis_result = await resp.json()
                    
                    # Store in session
                    cl.user_session.set("current_report", analysis_result)
                    
                    # Format and send response
                    response = format_analysis_response(analysis_result)
                    await cl.Message(content=response).send()
                else:
                    error_msg = await resp.text()
                    await cl.Message(content=f"分析失败: {error_msg}").send()
    
    except Exception as e:
        await cl.Message(content=f"处理失败: {str(e)}").send()
    
    finally:
        await processing_msg.remove()


async def handle_question(message: cl.Message):
    """Handle user questions"""
    
    question = message.content
    
    # Show thinking message
    thinking_msg = cl.Message(content="正在思考...")
    await thinking_msg.send()
    
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "question": question,
                "user_id": cl.user_session.get("user_id")
            }
            
            async with session.post(f"{API_URL}/api/ask-question", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    
                    # Format response
                    answer = result['answer']
                    validation = result.get('validation', {})
                    
                    if validation.get('warnings'):
                        answer += "\n\n⚠️ 注意事项:\n"
                        for warning in validation['warnings']:
                            answer += f"- {warning}\n"
                    
                    await cl.Message(content=answer).send()
                else:
                    error_msg = await resp.text()
                    await cl.Message(content=f"回答生成失败: {error_msg}").send()
    
    except Exception as e:
        await cl.Message(content=f"处理问题时出错: {str(e)}").send()
    
    finally:
        await thinking_msg.remove()


def format_analysis_response(analysis_result):
    """Format analysis result for display"""
    
    response_parts = ["# 体检报告分析结果\n"]
    
    # Structured data
    structured = analysis_result.get('structured_data', {})
    if structured.get('tests'):
        response_parts.append("## 检测项目\n")
        for test in structured['tests'][:10]:  # Show first 10
            name = test.get('name', 'N/A')
            value = test.get('value', 'N/A')
            unit = test.get('unit', '')
            ref_range = test.get('reference_range', '')
            
            line = f"- **{name}**: {value} {unit}"
            if ref_range:
                line += f" (参考: {ref_range})"
            response_parts.append(line + "\n")
        response_parts.append("\n")
    
    # Risk assessment
    risk = analysis_result.get('risk_assessment', {})
    if risk.get('identified_risks'):
        response_parts.append("## ⚠️ 风险评估\n")
        response_parts.append(f"整体风险等级: **{risk.get('overall_risk', 'N/A')}**\n\n")
        
        for risk_item in risk['identified_risks'][:5]:
            response_parts.append(
                f"- {risk_item['test']}: {risk_item['value']} "
                f"(正常值: <{risk_item['threshold']}) - "
                f"严重程度: {risk_item['severity']}\n"
            )
        response_parts.append("\n")
    
    # Analysis
    analysis = analysis_result.get('analysis', '')
    if analysis:
        response_parts.append("## 专业分析\n")
        response_parts.append(analysis + "\n")
    
    # Footer
    response_parts.append("\n---\n")
    response_parts.append("💡 您可以继续提问关于报告的任何问题。")
    
    return ''.join(response_parts)


if __name__ == "__main__":
    import chainlit.cli
    chainlit.cli.run_chainlit(__file__)
