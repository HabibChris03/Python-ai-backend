# DocFinder AI Backend

AI-powered backend for document analysis, image recognition, and chatbot functionality.

## Features

- **Document OCR**: Extract text from images using Tesseract and EasyOCR
- **Image Recognition**: Identify document types using CLIP model
- **Document Analysis**: Complete analysis including OCR, recognition, and content analysis
- **AI Chatbot**: Document management assistant using OpenAI GPT

## Installation

1. Install Python dependencies:

```bash
cd ai_backend
pip install -r requirements.txt
```

2. Install Tesseract OCR:

- **Windows**: Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

Start the server:

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### OCR (`POST /api/ocr`)

Extract text from uploaded images.

**Request**: Multipart form with image file
**Response**: OCR result with text, confidence, and bounding boxes

### Image Recognition (`POST /api/image-recognition`)

Recognize document type and extract features.

**Request**: Multipart form with image file
**Response**: Document type, confidence, description, and key features

### Document Analysis (`POST /api/document-analysis`)

Complete document analysis including OCR, recognition, and content analysis.

**Request**: Multipart form with image file
**Response**: Complete analysis with OCR results, recognition results, summary, and keywords

### Chatbot (`POST /api/chat`)

Chat with document management AI assistant.

**Request**: Form data with message and optional context
**Response**: AI response, detected intent, and confidence

### Health Check (`GET /api/health`)

Check server and model status.

## Models Used

- **CLIP**: For image recognition and document type classification
- **EasyOCR**: For text extraction
- **Tesseract**: Fallback OCR engine
- **Sentence Transformers**: For content analysis
- **OpenAI GPT**: For chatbot functionality

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `TESSERACT_PATH`: Path to Tesseract executable (optional)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: true)

## Example Usage

### OCR Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ocr",
    files={"file": open("document.jpg", "rb")}
)
print(response.json())
```

### Chat Example

```python
response = requests.post(
    "http://localhost:8000/api/chat",
    data={"message": "How do I organize my receipts?", "context": "user has 50 receipts"}
)
print(response.json())
```

## Performance Notes

- Models are loaded once at startup for optimal performance
- GPU acceleration is used when available
- EasyOCR is preferred over Tesseract for better accuracy
- CLIP model provides robust document type recognition

## Security

- CORS is enabled for all origins (configure for production)
- File validation ensures only images are processed
- OpenAI API key should be kept secure
