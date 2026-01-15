# Mr.🆖 DiscussAI 🎙️🎧

<p align="center">
  <strong>AI Speaking Tutor for Hong Kong Students 🇭🇰</strong>
</p>

<p align="center">
  Enhance speaking skills through AI-generated group discussions. Transform topics into realistic 4-student dialogues with authentic conversation strategies for HKDSE oral exam preparation.
</p>

## Features

- **🎓 Educational Focus:** Designed specifically for Hong Kong secondary students to practice group discussion skills for HKDSE oral exams
- **👥 4-Student Dialogues:** Generates realistic conversations between Candidates A, B, C, and D with authentic interaction patterns
- **🗣️ Communication Strategies:** Incorporates essential discussion skills - initiating, maintaining, transitioning, responding, rephrasing, and clarifying ideas
- **📁 Multiple Input Methods:** Upload documents (PDF, DOCX), images with OCR (JPG, JPEG, PNG), or paste discussion topics directly
- **🤖 AI-Powered Generation:** Uses OpenAI's top-tier LLMs to create engaging, topic-relevant dialogues with proper brainstorming and structure
- **🎵 Audio Narration:** High-quality text-to-speech with natural voices through Mr.🆖 AI Hub for immersive practice sessions
- **🌈 Color-Coded Transcripts:** Each candidate's dialogue is displayed in a distinct color bubble for easy reading and review
- **📄 Word Document Export:** Download complete study materials as formatted Word documents with color-coded transcripts and structured notes
- **📚 Comprehensive Study Notes:** Every session generates structured learning materials including:
  - Ideas outline with Traditional Chinese translations
  - Vocabulary table with English terms, Chinese meanings, and usage examples
  - Communication strategies with HKDSE-relevant patterns and examples
- **⏱️ Exam-Length Practice:** Generates 6-9 minute discussions matching typical HKDSE oral exam duration
- **🎚️ Two Dialogue Modes:**
  - **Normal Mode:** Clear, exam-style conversations (6-7 minutes) for quick practice
  - **Deeper Mode:** More detailed discussions with further ideas and elaborations (7-9 minutes)
- **💰 Cost Transparency:** Real-time TTS cost calculation and tracking
- **💾 Practice History:** Save and review previous discussion sessions with full transcripts and audio in your browser

## How It Works

1. **Choose Input Method:**
   - Upload HKDSE exam papers or study materials (PDF, DOCX, images)
   - Or enter your own discussion topic directly

2. **Select Dialogue Depth:**
   - **Normal:** Standard exam-style discussions for efficient practice
   - **Deeper:** Extended discussions with more examples and elaboration for comprehensive review

3. **Enter API Key:**
   - Get your API key from [Mr.🆖 AI Hub](https://api.mr5ai.com)
   - Or set it as an environment variable

4. **Generate & Listen:**
   - AI generates authentic 4-student group discussion
   - Each candidate has a unique voice (powered by OpenAI TTS)
   - View color-coded transcript with speaker bubbles
   - Review comprehensive study notes below the transcript

5. **Download as Word Document:**
   - Click "Download as Word Document" to get a formatted .docx file
   - Includes color-coded transcript and all study notes
   - Perfect for offline study and printing

6. **Review History:**
   - All sessions are saved in your browser
   - Access previous discussions anytime from the Archives section

## Output Components

Every generation includes:

### 🎙️ Audio Discussion
- Authentic HKDSE-style group discussion
- 4 distinct AI voices
- 6-9 minutes duration matching real exam length
- Natural conversation flow with proper pacing

### 📃 Color-Coded Transcript
- Speaker bubbles in distinct colors:
  - Candidate A: Light Blue
  - Candidate B: Light Yellow
  - Candidate C: Light Green
  - Candidate D: Light Pink
- Easy to follow and review

### 📄 Word Document (Downloadable)
- Formatted .docx file with:
  - Color-coded transcript with background shading
  - Structured study notes (Ideas, Language, Communication Strategies)
  - Tables for vocabulary
  - Numbered lists for communication strategies

### 📚 Study Notes (English + Traditional Chinese)

**1. Ideas Section (💡 討論要點)**
- Structured outline of main discussion points
- Shows how each question prompt was addressed
- Traditional Chinese translations for all key concepts
- Clear hierarchy with main points and sub-points

**2. Language Section (📖 語言學習)**
- 10-15 useful vocabulary words from the dialogue
- English terms with Traditional Chinese translations
- Usage examples from the actual discussion
- Formatted as an easy-to-read table

**3. Communication Strategies Section (💬 溝通策略)**
- 6-10 interaction strategies demonstrated in the dialogue
- Real examples from the generated discussion
- Traditional Chinese explanations
- HKDSE-relevant patterns including:
  - Initiating discussion (開始討論)
  - Maintaining discussion (維持討論)
  - Transitioning between topics (轉換話題)
  - Responding and agreeing/disagreeing (回應及表達同意/不同意)
  - Asking for clarification (要求澄清)
  - Rephrasing (重新表述)
  - Summarizing (總結)
  - Elaborating with examples (舉例說明) - in Deeper mode
  - Building on others' ideas (延伸他人想法) - in Deeper mode

## Demo Examples

The project includes sample HKDSE exam materials:

- **DSE 2023 Paper 4 Set 1.2.png** - Sample group discussion exam paper for practice
- **DSE 2024 Paper 4 Set 5.2.png** - Additional HKDSE oral exam materials

These examples demonstrate how to transform exam topics into interactive group discussions for oral practice.

## Dialogue Modes Comparison

| Feature | Normal Mode | Deeper Mode |
|---------|-------------|-------------|
| Duration | 6-7 minutes | 7-9 minutes |
| Detail Level | Clear, focused | More elaboration |
| Examples | Key examples | Multiple examples per point |
| Best For | Quick practice, time efficiency | Comprehensive review, vocabulary building |
| Learning Notes | 6 strategies | 6-8 strategies |

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
   export OPENAI_MODEL_NORMAL="gpt-5-mini"  # Model for Normal mode (optional)
   export OPENAI_MODEL_DEEP="gpt-5.1"    # Model for Deeper mode (optional)
   ```

   - Optional: Configure Sentry for error monitoring:

   ```bash
   export SENTRY_DSN="your-sentry-dsn"  # Optional
   ```

## Usage

### Run the Application

```bash
# Using uv (recommended)
uv run python main.py

# Or with standard Python
python main.py
```

The application will start on `http://0.0.0.0:8000`

### Using the Web Interface

1. Open your browser to `http://localhost:8000`
2. Choose your input method:
   - **Upload Files:** Drag and drop HKDSE papers or study materials
   - **Enter Topic:** Type or paste discussion topics directly
3. Select dialogue depth (Normal or Deeper)
4. Enter your Mr.🆖 AI Hub API key (if not set in environment)
5. Click "Generate Discussion with Audio and Study Notes"
6. Listen to the audio, read the color-coded transcript, and review study notes
7. Click "Download as Word Document" to get a formatted .docx file
8. Access previous sessions in the Archives section

## Technical Features

- **Concurrent Audio Generation:** Uses ThreadPoolExecutor for parallel TTS processing (10 workers)
- **Retry Mechanisms:** Built-in retry logic with exponential backoff for API calls (3 attempts)
- **Multiple File Format Support:**
  - PDF: Text extraction with PyPDF
  - DOCX: Document parsing with python-docx
  - Images: OCR with OpenAI Vision API (GPT-4.1-mini)
- **Word Document Export:** Generates formatted .docx files with color-coded transcripts and structured notes
- **Cost Tracking:** Real-time TTS cost calculation in HKD
- **Browser-Based Storage:** Practice history saved in localStorage
- **Automatic Cleanup:** Old temporary audio files removed after 24 hours, Word documents after 60 minutes
- **Timezone Support:** Timestamps in Hong Kong timezone (Asia/Hong_Kong)
- **Error Handling:** Comprehensive error handling with detailed logging via Loguru
- **Sentry Integration:** Optional error monitoring via SENTRY_DSN environment variable

## API Integration

This application uses Mr.🆖 AI Hub's OpenAI-compatible API:

- **Dialogue Generation:** Configurable models (default: GPT-5) with structured output (Pydantic models)
- **Text-to-Speech:** OpenAI TTS-1 with 4 distinct voices (nova, alloy, fable, echo)
- **Vision OCR:** GPT-4.1-mini vision model for image text extraction

## File Structure

```
discussAI/
├── main.py                 # Main application file
├── description.md          # App description for UI
├── footer.md               # Footer content for UI
├── head.html               # Custom HTML/JS for browser storage
├── examples/               # Sample HKDSE exam papers
│   ├── DSE 2023 Paper 4 Set 1.2.png
│   └── DSE 2024 Paper 4 Set 5.2.png
├── static/                 # Static assets (logo, icon)
│   ├── logo.png
│   └── icon.png
├── gradio_cached_files/    # Temporary audio and document files
│   └── tmp/
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Dependencies

Key dependencies:
- `gradio` - Web UI framework
- `openai` - OpenAI API client
- `promptic` - LLM decorator for structured outputs
- `pydantic` - Data validation and structured models
- `pypdf` - PDF text extraction
- `python-docx` - DOCX document parsing and Word document generation
- `tenacity` - Retry mechanisms
- `fastapi` - Web framework
- `loguru` - Logging
- `pytz` - Timezone support
- `requests` - HTTP requests for TTS
- `beautifulsoup4` - HTML parsing for Word document conversion

## Troubleshooting

### Common Issues for Students & Teachers

- **API Key Issues:** Ensure your Mr.🆖 AI Hub key is valid and has sufficient credits for TTS generation
- **Study Material Upload:** Check that HKDSE papers and study materials are in supported formats (PDF, DOCX, JPG, JPEG, PNG)
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

**Made with ❤️ for Hong Kong students preparing for HKDSE oral exams**
