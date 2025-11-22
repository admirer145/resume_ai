# Resume Intelligence Analysis System - Ollama Setup

## Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)

2. **Pull Required Models**:
```bash
# Pull embedding model
ollama pull embeddinggemma:latest

# Pull LLM model
ollama pull deepseek-coder:1.3b

# Optional: Pull larger models for better quality
# ollama pull llama3.2:3b
# ollama pull qwen2.5:7b
# ollama pull mistral:7b
```

3. **Start Ollama Server**:
```bash
ollama serve
```
The server will run on `http://localhost:11434` by default.

4. **Install Weaviate** (for vector database):
```bash
docker run -d \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  --name weaviate \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

## Backend Setup

1. **Install Dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Configure Environment**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env file (already configured for Ollama by default)
# No changes needed if using default Ollama setup
```

3. **Start Backend**:
```bash
uvicorn main:app --reload --port 8000
```

## Frontend Setup

1. **Install Dependencies**:
```bash
cd resume-ui
npm install
```

2. **Start Frontend**:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Upload a resume PDF
3. Paste a job description
4. Click "Analyze Resume"
5. View the detailed analysis with scores and suggestions

## Switching Models

To use different models, edit `.env`:

```bash
# Use larger LLM for better quality
LLM_MODEL=llama3.2:3b

# Or use OpenAI instead
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
```

## Model Recommendations

### For Low-Resource Systems (Your Current Setup):
- **Embedding**: `embeddinggemma:latest` ✅
- **LLM**: `deepseek-coder:1.3b` ✅

### For Better Quality (if system allows):
- **LLM**: `llama3.2:3b` (3GB RAM) - Better reasoning
- **LLM**: `qwen2.5:7b` (4.7GB RAM) - Best balance
- **LLM**: `mistral:7b` (4.1GB RAM) - Strong instruction following

### For Production (with budget):
- **Provider**: OpenAI
- **Embedding**: `text-embedding-3-small`
- **LLM**: `gpt-4o-mini` or `gpt-4`

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Weaviate Connection Error
```bash
# Check if Weaviate is running
docker ps | grep weaviate

# Restart Weaviate
docker restart weaviate
```

### Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull <model-name>
```

### JSON Parsing Errors
If you get JSON parsing errors with smaller models, try:
1. Using a larger model like `llama3.2:3b`
2. Reducing resume/JD text length in prompts
3. Increasing `num_predict` in `llm_analyzer.py`

## Performance Notes

- **deepseek-coder:1.3b**: Fast but may produce less accurate analysis
- **llama3.2:3b**: Good balance of speed and quality
- **qwen2.5:7b**: Best quality but slower
- Analysis time: 30-60 seconds per resume (depending on model)
