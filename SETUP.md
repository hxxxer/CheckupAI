# CheckupAI Setup Guide

## Prerequisites

- Python 3.8+
- Docker (for Milvus)
- GPU (optional, for faster inference)

## Step-by-Step Setup

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd CheckupAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Milvus Vector Database

Using Docker (Recommended):

```bash
# Pull and run Milvus standalone
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest

# Verify Milvus is running
docker ps
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Required configurations:
# - MILVUS_HOST=localhost
# - MILVUS_PORT=19530
# - OPENAI_API_KEY=your_key (if using OpenAI for validation)
```

### 4. Download Pre-trained Models

```bash
# Download BERT model for NER (automatic on first run)
# Download Qwen3-8B model (or your chosen LLM)
# Place fine-tuned model in: backend/model/lora_checkpoint/
```

### 5. Build Knowledge Base

```bash
# Add medical documents to data/knowledge_base/raw/
# Supported formats: PDF, DOC, DOCX

# Run knowledge base builder
python scripts/build_knowledge_base.py
```

### 6. Start Backend API

```bash
# Run API server
cd backend/api
python main.py

# API will be available at: http://localhost:8000
# Check health: curl http://localhost:8000/api/health
```

### 7. Start Frontend

```bash
# Open new terminal
cd frontend

# Run Chainlit app
chainlit run app.py

# Frontend will be available at: http://localhost:8080
```

## Verification

1. Open browser to http://localhost:8080
2. Upload a test medical report image
3. Verify the analysis is generated

## Troubleshooting

### Milvus Connection Error

```bash
# Check Milvus is running
docker ps | grep milvus

# Check logs
docker logs milvus-standalone

# Restart if needed
docker restart milvus-standalone
```

### OCR Issues

```bash
# Install PaddleOCR dependencies
pip install paddlepaddle paddleocr

# For GPU support
pip install paddlepaddle-gpu
```

### Model Loading Issues

- Ensure model path in .env is correct
- Check if model files exist in the specified directory
- Verify sufficient memory/VRAM for model loading

### API Connection Issues

- Verify backend API is running on port 8000
- Check API_URL in .env matches the backend address
- Test API endpoint: `curl http://localhost:8000/api/health`

## Production Deployment

### Security Considerations

1. Enable authentication for API endpoints
2. Use HTTPS for all communications
3. Encrypt sensitive data at rest
4. Set up proper firewall rules
5. Use environment variables for secrets (never commit .env)

### Performance Optimization

1. Use GPU for model inference
2. Enable Milvus indexing for faster search
3. Implement caching for frequently accessed data
4. Use load balancer for multiple API instances

### Monitoring

1. Set up logging and monitoring
2. Monitor API response times
3. Track Milvus query performance
4. Monitor model inference latency

## Next Steps

- Fine-tune the model with your medical data
- Add more documents to knowledge base
- Customize risk assessment rules
- Implement user authentication
- Add multi-language support

## Support

For issues, please check:
1. Logs in backend API
2. Milvus container logs
3. Frontend console errors

For questions, open an issue on the repository.
