"""
API Module for CheckupAI
FastAPI endpoints for frontend integration
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import json

# Import backend modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ocr.paddle_table_parser import PaddleTableParser
from ocr.handwrite_enhancer import HandwriteEnhancer
from ner.medical_ner import MedicalNER
from ner.llm_validator import LLMValidator
from rag.retriever import DualPathRetriever
from rag.user_profile import UserProfileManager
from rag.risk_guard import RiskGuard
from model.inference import MedicalLLMInference


app = FastAPI(title="CheckupAI API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ReportAnalysisRequest(BaseModel):
    user_id: Optional[str] = None
    image_path: str


class QuestionRequest(BaseModel):
    user_id: Optional[str] = None
    question: str


class AnalysisResponse(BaseModel):
    structured_data: dict
    analysis: str
    risk_assessment: dict
    recommendations: List[str]


# Initialize components (lazy loading)
ocr_parser = None
ner_model = None
retriever = None
llm_inference = None
risk_guard = None


def get_ocr_parser():
    global ocr_parser
    if ocr_parser is None:
        ocr_parser = PaddleTableParser()
    return ocr_parser


def get_ner_model():
    global ner_model
    if ner_model is None:
        ner_model = MedicalNER()
    return ner_model


def get_retriever():
    global retriever
    if retriever is None:
        retriever = DualPathRetriever()
    return retriever


def get_llm_inference():
    global llm_inference
    if llm_inference is None:
        model_path = os.getenv('MODEL_PATH', './model/lora_checkpoint')
        llm_inference = MedicalLLMInference(model_path)
    return llm_inference


def get_risk_guard():
    global risk_guard
    if risk_guard is None:
        risk_guard = RiskGuard()
    return risk_guard


@app.get("/")
def read_root():
    return {"message": "CheckupAI API is running"}


@app.post("/api/upload-report")
async def upload_report(file: UploadFile = File(...), user_id: Optional[str] = None):
    """
    Upload and process medical report
    """
    # Save uploaded file
    upload_dir = "../../data/sensitive/raw_reports"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "message": "File uploaded successfully",
        "file_path": file_path,
        "user_id": user_id
    }


@app.post("/api/analyze-report", response_model=AnalysisResponse)
async def analyze_report(request: ReportAnalysisRequest):
    """
    Analyze medical report
    """
    try:
        # Step 1: OCR
        parser = get_ocr_parser()
        parsed_data = parser.parse_table(request.image_path)
        
        # Step 2: NER
        ner = get_ner_model()
        structured_data = ner.structure_report(' '.join(parsed_data['content']))
        
        # Step 3: RAG retrieval
        retriever = get_retriever()
        context = retriever.dual_retrieve(
            query=' '.join(parsed_data['content']),
            user_id=request.user_id
        )
        
        # Step 4: LLM inference
        llm = get_llm_inference()
        analysis = llm.generate_report_analysis(
            structured_data,
            context=context
        )
        
        # Step 5: Risk assessment
        guard = get_risk_guard()
        risk_assessment = guard.assess_risk_level(structured_data)
        
        return AnalysisResponse(
            structured_data=structured_data,
            analysis=analysis,
            risk_assessment=risk_assessment,
            recommendations=[]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask-question")
async def ask_question(request: QuestionRequest):
    """
    Answer user question with RAG
    """
    try:
        # Retrieve context
        retriever = get_retriever()
        context = retriever.dual_retrieve(
            query=request.question,
            user_id=request.user_id
        )
        
        # Generate answer
        llm = get_llm_inference()
        answer = llm.generate_qa_response(request.question, context)
        
        # Validate answer
        guard = get_risk_guard()
        validation = guard.validate_recommendation(answer, {})
        
        return {
            "question": request.question,
            "answer": answer,
            "validation": validation,
            "context_sources": len(context.get('knowledge', []))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
