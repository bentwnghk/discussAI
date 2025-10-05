# Mr.🆖 DiscussAI 🎙️🎧

<p align="center">
  <strong>Transform documents, images, and websites into engaging podcast episodes using AI</strong>
</p>

<p align="center">
  Convert long-form content into natural podcast dialogues that capture attention and make information more accessible for auditory learning on the go.
</p>

## Features

- **📁 Versatile Input Support:** Upload documents (PDF, DOCX, TXT), images with OCR (JPG, JPEG, PNG), paste text directly, or convert websites via URL
- **🤖 AI-Powered Dialogue Generation:** Uses OpenAI's GPT-4.1-mini to create natural, engaging podcast conversations from your content
- **🎵 Professional Audio:** Leverages OpenAI's text-to-speech models for high-quality, lifelike voices through Mr.🆖 AI Hub routing
- **🌐 Multi-Language Support:** Generate podcasts in English, Chinese (Traditional), or Cantonese with optimized voice synthesis
- **💰 Cost Transparency:** Real-time TTS cost calculation and tracking (English, Chinese, Cantonese)
- **🖥️ User-Friendly Interface:** Gradio-based web interface for easy interaction
- **💾 Smart History Management:** Browse and reload previous podcasts stored in your browser (IndexedDB + localStorage)
- **🔧 Resilient Processing:** Retry mechanisms and error handling for reliable conversion
- **⚡ FastAPI Backend:** Robust server architecture with deployment-ready setup

## Demo Examples

The project includes sample inputs:
- PDF documents (e.g., "Intangible cultural heritage item.pdf")
- Images with text (e.g., "JUPAS Guide.jpg")
- URL extraction from web pages

## Installation

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bentwnghk/mr5-podcast-ai.git
   cd mr5-podcast-ai
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Environment Setup:**
   - Get an API key from [Mr.🆖 AI Hub](https://api.mr5ai.com)
   - Set environment variables:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     export OPENAI_BASE_URL="https://api.mr5ai.com/v1"  # Mr.🆖 AI Hub endpoint
     ```
   - Optional: Configure Sentry for error monitoring:
     ```bash
     export SENTRY_DSN="your-sentry-dsn"
     ```

## Usage

### Quick Start

1. **Launch the Application:**
   ```bash
   uv run python main.py
   ```
   The Gradio interface will open in your browser at http://localhost:8000

2. **Generate Podcasts:**
   - **Upload Files:** Select PDF, DOCX, TXT, or image files
   - **Paste Text:** Directly input text content
   - **From URL:** Convert web pages to podcasts
   - Choose language: English, Chinese, or Cantonese
   - Enter your API key (auto-saved to browser)
   - Click "Generate Podcast"

3. **View Results:**
   - Listen to the generated MP3 podcast
   - Review the dialogue transcript
   - Check TTS costs
   - Access previous podcasts in the History panel

### Input Methods

#### 1. File Upload
- **PDF:** Digital documents, scans, reports
- **DOCX:** Word documents
- **TXT:** Plain text files
- **Images:** JPG/PNG with text extraction via OpenAI Vision API

#### 2. Text Input
- Paste any text content
- Supports up to ~8000 token dialogues

#### 3. URL Processing
- Convert web articles to podcasts
- Automatic content extraction with fallbacks
- Supports major news sites and blogs

### Language Options

- **English:** Standard OpenAI TTS models
- **Chinese (繁體):** Traditional Chinese with optimized output
- **Cantonese:** Specialized voice support through Mr.🆖 AI Hub

## Architecture

- **Frontend:** Gradio web interface
- **Backend:** FastAPI server with async processing
- **Storage:** Temporary file management with auto-cleanup
- **AI Services:** OpenAI GPT-4.1-mini + TTS via Mr.🆖 AI Hub
- **Database:** Browser-based history (IndexedDB/localStorage)
- **Deployment:** Ready for Docker/Uvicorn

## Cost Estimation

TTS costs vary by language:
- English: ~$0.15 per 1M characters
- Chinese: ~$0.30 per 1M characters (x2 multiplier)
- Cantonese: ~$0.75 per 1M characters (x8 multiplier)

## Project Structure

```
.
├── main.py              # Application entry point
├── description.md        # UI descriptions
├── head.html             # Custom HTML/JS for browser features
├── static/               # Web assets (logo, icon)
├── examples/             # Sample files for testing
├── pyproject.toml        # Python dependencies
├── uv.lock              # Dependency lock file
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Docker composition
├── LICENSE              # Apache 2.0 License
└── README.md            # This file
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Mr.🆖 AI Hub API key | Yes |
| `OPENAI_BASE_URL` | Mr.🆖 AI Hub endpoint URL | Yes |
| `SENTRY_DSN` | Sentry monitoring DSN | No |

### Custom API Endpoints

The application is designed to work with Mr.🆖 AI Hub compatible endpoints. Set `OPENAI_BASE_URL` to:
- Production: `https://api.mr5ai.com/v1`
- Local: `http://localhost:3000/v1` (if running locally)

## Troubleshooting

### Common Issues

- **API Key Issues:** Ensure your Mr.🆖 AI Hub key is valid and has sufficient credits
- **File Upload Errors:** Check file size limits and supported formats
- **URL Processing:** Some websites block scraping - try different sources
- **TTS Failures:** Request timeouts - the app has retry mechanisms

### Debug Mode

Set `python -c "import logging; logging.basicConfig(level=logging.DEBUG)"` before launching for detailed logs.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Support

- Create issues on [GitHub](https://github.com/bentwnghk/mr5-podcast-ai) for bugs/feature requests
- Check the examples directory for sample inputs
- Review browser console for detailed error messages

---

<div align="center">
  Transform your content into <strong>podcasts that engage and inform</strong> 🎙️
</div>
