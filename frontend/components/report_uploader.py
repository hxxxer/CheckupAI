"""
Report Uploader Component
Custom UI component for file upload
"""

import chainlit as cl


class ReportUploader:
    @staticmethod
    async def create_upload_button():
        """Create file upload button"""
        
        elements = [
            cl.File(
                name="medical_report",
                display="inline",
                accept=["image/png", "image/jpeg", "image/jpg"]
            )
        ]
        
        return elements
    
    @staticmethod
    async def show_upload_prompt():
        """Show upload prompt to user"""
        
        await cl.Message(
            content="请上传体检报告图片 (支持 PNG, JPG, JPEG 格式)"
        ).send()
