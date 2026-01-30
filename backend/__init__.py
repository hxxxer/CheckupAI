# CheckupAI Backend

AI-powered medical checkup report analysis system with RAG and LLM.

## Project Structure

```
CheckupAI/
├── backend/                   # Backend core
│   ├── ocr/                   # OCR module
│   ├── ner/                   # NER and structuring
│   ├── rag/                   # RAG system
│   ├── model/                 # Model inference
│   ├── api/                   # API endpoints
│   └── config/                # Configuration
├── frontend/                  # Chainlit frontend
├── data/                      # Data directory
│   ├── sensitive/             # Sensitive data (not tracked)
│   ├── knowledge_base/        # Medical knowledge base
│   └── finetune_data/         # LoRA fine-tuning data
├── scripts/                   # Utility scripts
└── requirements.txt
```

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - Set MILVUS_HOST and MILVUS_PORT
# - Set OPENAI_API_KEY if using OpenAI
# - Configure MODEL_PATH for your fine-tuned model
```

### 3. Setup Milvus

```bash
# Using Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### 4. Build Knowledge Base

```bash
# Place medical documents in data/knowledge_base/raw/
# Then run the build script
python scripts/build_knowledge_base.py
```

### 5. Run Backend API

```bash
cd backend/api
python main.py

# API will be available at http://localhost:8000
```

### 6. Run Frontend

```bash
cd frontend
chainlit run app.py

# Frontend will be available at http://localhost:8080
```

## Module Overview

### OCR Module
- **paddle_table_parser.py**: Table structure recognition using PaddleOCR
- **handwrite_enhancer.py**: Handwriting enhancement for better OCR accuracy

### NER Module
- **medical_ner.py**: BERT-based named entity recognition for medical reports
- **llm_validator.py**: LLM-based validation and completion of NER results

### RAG Module
- **retriever.py**: Dual-path retrieval (knowledge base + user profile)
- **user_profile.py**: User health profile management
- **risk_guard.py**: Rule-based validation layer for safety

### Model Module
- **inference.py**: Fine-tuned Qwen3-8B inference
- **lora_finetune/train.py**: LoRA fine-tuning script

### API Module
- **main.py**: FastAPI endpoints for frontend integration

## API Endpoints

- `POST /api/upload-report`: Upload medical report image
- `POST /api/analyze-report`: Analyze medical report
- `POST /api/ask-question`: Ask questions about health data
- `GET /api/health`: Health check endpoint

## Fine-tuning

To fine-tune the model:

1. Prepare training data in `data/finetune_data/train.json`
2. Run the fine-tuning script:

```bash
cd backend/model/lora_finetune
python train.py
```

## Scripts

- **build_knowledge_base.py**: Build vector database from medical documents
- **profile_sync.py**: Sync user profiles to Milvus

## Security Notes

- All sensitive data (reports, personal info) is stored in `data/sensitive/` which is excluded from git
- Never commit actual medical reports or personal information
- Use strong authentication in production
- Encrypt data at rest and in transit

## Development

### Adding New Features

1. Backend modules: Add to appropriate directory under `backend/`
2. API endpoints: Update `backend/api/main.py`
3. Frontend components: Add to `frontend/components/`

### Testing

```bash
# Run tests (add your test framework)
pytest tests/
```

## License

This project is for educational and research purposes only. Consult medical professionals for actual health advice.

## Support

For issues and questions, please open an issue on the repository.
