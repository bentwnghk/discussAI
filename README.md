# Mr.🆖 DiscussAI 🎙️🎧

<p align="center">
  <strong>AI Language Tutor for Hong Kong Students 🇭🇰</strong>
</p>

<p align="center">
  Enhance speaking skills through AI-generated group discussions. Transform topics into realistic 4-student dialogues with authentic conversation strategies for HKDSE oral exam preparation.
</p>

## Features

- **🎓 Educational Focus:** Designed specifically for Hong Kong secondary students to practice group discussion skills for HKDSE oral exams
- **👥 4-Student Dialogues:** Generates realistic conversations between Candidates A, B, C, and D with authentic interaction patterns
- **🗣️ Communication Strategies:** Incorporates essential discussion skills - initiating, maintaining, responding, and rephrasing ideas
- **📁 Multiple Input Methods:** Upload documents (PDF, DOCX, TXT), images with OCR (JPG, JPEG, PNG), or paste discussion topics directly
- **🤖 AI-Powered Generation:** Uses OpenAI's GPT-4.1-mini to create engaging, topic-relevant dialogues with proper brainstorming and structure
- **🎵 Audio Narration:** High-quality text-to-speech with natural voices through Mr.🆖 AI Hub for immersive practice sessions
- **🌐 Multi-Language Support:** Practice in English, Chinese (Traditional), or Cantonese - perfect for Hong Kong's linguistic diversity
- **⏱️ Exam-Length Practice:** Generates 8-10 minute discussions matching typical HKDSE oral exam duration
- **💰 Cost Transparency:** Real-time TTS cost calculation and tracking for different languages
- **💾 Practice History:** Save and review previous discussion sessions with full transcripts and audio

## Demo Examples

The project includes sample HKDSE exam materials:
- **DSE 2019 Paper 4 Set 2.2.png** - Sample group discussion exam paper for practice
- **DSE 2023 Paper 4 Set 1.1.png** - Additional HKDSE oral exam materials

These examples demonstrate how to transform exam topics into interactive group discussions for oral practice.

## Installation

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bentwnghk/discussAI.git
   cd discussAI
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

2. **Generate Group Discussions:**
   - **Upload HKDSE Materials:** Use sample exam papers or upload your own study materials (PDF, DOCX, TXT, or images)
   - **Enter Discussion Topics:** Paste exam topics or practice questions directly
   - **Select Language:** Choose English, Chinese (Traditional), or Cantonese for practice
   - **Enter API Key:** Add your Mr.🆖 AI Hub API key (auto-saved to browser)
   - **Click "Generate Discussion"**

3. **Practice & Review:**
   - **Listen to Audio:** Hear realistic 4-student discussions with natural conversation flow
   - **Study Transcripts:** Review dialogue structure and communication strategies
   - **Track Costs:** Monitor TTS expenses for different languages
   - **Build Practice History:** Save and revisit previous sessions for continued improvement

### Input Methods

#### 1. HKDSE Study Materials
- **PDF:** Exam papers, study guides, and practice materials
- **DOCX:** Word documents with discussion topics and questions
- **TXT:** Practice questions and topic outlines
- **Images:** Scanned exam papers and study materials (JPG/PNG with OCR)

#### 2. Direct Topic Input
- **Discussion Topics:** Paste exam questions or practice topics directly
- **Custom Scenarios:** Create your own discussion situations
- **Text Length:** Supports comprehensive topics for 8-10 minute dialogues

#### 3. Educational Content
- **Study Materials:** Convert any educational content into discussion practice
- **Exam Questions:** Transform test questions into interactive group discussions
- **Practice Scenarios:** Create realistic oral exam situations

### Language Options

Perfect for Hong Kong's multilingual environment:

- **English:** Practice for HKDSE English Language Paper 4 (School-based Assessment)
- **Chinese (繁體):** Traditional Chinese discussions for Chinese Language oral exams
- **Cantonese:** Native Hong Kong Cantonese for authentic local language practice

Each language includes culturally appropriate communication strategies and exam-relevant vocabulary.

## Architecture

**Designed for Educational Excellence:**

- **Frontend:** Intuitive Gradio web interface optimized for student learning
- **Backend:** FastAPI server with robust async processing for reliable generation
- **AI Engine:** OpenAI GPT-4.1-mini specialized for educational dialogue creation
- **Voice Synthesis:** Premium TTS through Mr.🆖 AI Hub with multi-language support
- **Storage:** Browser-based history system for tracking practice progress
- **Deployment:** Production-ready with Docker support for institutional use

## Educational Investment

**Transparent Pricing for Quality Learning:**

TTS costs vary by language and are clearly displayed after each generation:
- **English:** ~$0.15 per 1M characters (standard rate)
- **Chinese:** ~$0.30 per 1M characters (2x multiplier for Traditional Chinese)
- **Cantonese:** ~$0.75 per 1M characters (8x multiplier for native Hong Kong Cantonese)

*Typical 8-10 minute discussion: $0.05-0.25 depending on language choice*

## Project Structure

```
.
├── main.py              # Main application with AI dialogue generation
├── description.md        # Educational UI descriptions and messaging
├── head.html             # Advanced browser features for practice history
├── static/               # Web assets (logo, educational icons)
├── examples/             # HKDSE sample papers for demonstration
├── pyproject.toml        # Educational AI and web framework dependencies
├── uv.lock              # Locked dependency versions
├── Dockerfile           # Container for institutional deployment
├── docker-compose.yml   # Complete deployment configuration
├── LICENSE              # Apache 2.0 License
└── README.md            # Educational project documentation
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

### Common Issues for Students & Teachers

- **API Key Issues:** Ensure your Mr.🆖 AI Hub key is valid and has sufficient credits for TTS generation
- **Study Material Upload:** Check that HKDSE papers and study materials are in supported formats (PDF, DOCX, TXT, JPG, PNG)
- **Content Extraction:** Some scanned documents might need better quality images for accurate text extraction
- **Generation Timeouts:** Complex topics might take longer - the app has retry mechanisms built-in
- **Audio Quality:** Ensure stable internet connection for consistent TTS generation

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

- Create issues on [GitHub](https://github.com/bentwnghk/discussAI) for bugs/feature requests
- Check the examples directory for sample inputs
- Review browser console for detailed error messages

---

<div align="center">
  <strong>Empower Hong Kong students with AI-enhanced speaking practice</strong> 🇭🇰🗣️
</div>
