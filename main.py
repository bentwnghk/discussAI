# -*- coding: utf-8 -*-
import concurrent.futures as cf
import glob
import io
import os
import time
import datetime
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional, Dict, Any
import json
import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger # logger is imported here
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential # tenacity is imported here
from mimetypes import guess_type
import docx # Added for DOCX support
import pytz

OPENAI_VOICE_MAPPINGS = {
    "female-1": "nova",
    "male-1": "alloy",
    "female-2": "fable",
    "male-2": "echo",
}

if sentry_dsn := os.getenv("SENTRY_DSN"):
    sentry_sdk.init(sentry_dsn)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["female-1", "male-1", "female-2", "male-2"]

    def voice(self, language: str = "English"): # Add language parameter, remove @property
        # Always use OPENAI_VOICE_MAPPINGS.
        # one-api will be responsible for mapping these voice IDs (e.g., "nova")
        # and the 'language' from the payload to the correct downstream provider voice.
        return OPENAI_VOICE_MAPPINGS[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]


# Add retry mechanism to TTS calls for resilience
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def get_mp3(text: str, voice: str, api_key: str = None, language_selection: str = "English") -> bytes:
    """
    Generates MP3 audio for the given text by making a direct request to the
    one-api compatible endpoint (expected at OPENAI_BASE_URL).
    The 'language_selection' parameter is included in the payload to one-api,
    allowing one-api to route to appropriate downstream TTS providers (e.g., MiniMax)
    and apply necessary language boosts. Includes retries.
    """
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") # This should be your one-api base URL

    if not effective_api_key:
        logger.error("API key is not configured.")
        raise ValueError("API key not configured.")
    if not base_url:
        logger.error("Base URL is not configured.")
        raise ValueError("Base URL not configured.")

    # Construct the full URL to the one-api speech endpoint
    # Assumes one-api exposes an OpenAI-compatible speech endpoint at /v1/audio/speech
    speech_endpoint_url = f"{base_url.rstrip('/')}/audio/speech"

    headers = {
        "Authorization": f"Bearer {effective_api_key}",
        "Content-Type": "application/json"
    }

    # Manually construct the payload. The 'language': 'Cantonese' field is included
    # as per the user's requirement for one-api to process this for MiniMax language_boost.
    payload: Dict[str, Any] = { # Ensure payload is explicitly typed for clarity
        "voice": voice,
        "input": text,
        "response_format": "mp3",
    }

    if language_selection == "Cantonese":
        payload["model"] = "speech-2.5-turbo-preview"
        payload["language"] = language_selection # This field is intended for one-api
                                                 # to trigger MiniMax language_boost if needed.
    elif language_selection in ["English", "Chinese"]:
        payload["model"] = "tts-1"
        # tts-1 does not accept the language parameter, so it's omitted.
    else:
        # Default or fallback behavior if a new language is added without explicit handling
        logger.warning(f"Unhandled language_selection '{language_selection}', defaulting to speech-2.5-turbo-preview and including language parameter.")
        payload["model"] = "speech-2.5-turbo-preview"
        payload["language"] = language_selection

    logger.debug(
        f"Requesting TTS. Endpoint: '{speech_endpoint_url}', "
        f"Voice: '{voice}', Selected Language for Payload: '{language_selection}', Text: '{text[:50]}...'"
    )

    try:
        response = requests.post(
            speech_endpoint_url,
            headers=headers,
            json=payload,
            timeout=60.0 # Timeout in seconds
        )
        response.raise_for_status() # Raise an exception for HTTP error codes (4xx or 5xx)

        logger.debug(
            f"TTS generation successful. Voice: '{voice}', Text: '{text[:50]}...'"
        )
        return response.content  # The binary content of the MP3

    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error during TTS generation. Voice: '{voice}', Text: '{text[:50]}...'. " \
                        f"Status: {http_err.response.status_code if http_err.response else 'N/A'}. " \
                        f"Error: {http_err}. " \
                        f"Response: {http_err.response.text if http_err.response else 'No response text'}"
        logger.error(error_message)
        raise # Reraise exception to trigger tenacity retry
    except requests.exceptions.RequestException as req_err: # Catches other requests errors (e.g., timeout, connection error)
        logger.error(
            f"Request error during TTS generation. Voice: '{voice}', Text: '{text[:50]}...'. Error: {req_err}"
        )
        raise # Reraise exception to trigger tenacity retry
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(
            f"Unexpected error during TTS generation. Voice: '{voice}', Text: '{text[:50]}...'. Error: {e}"
        )
        raise # Reraise exception to trigger tenacity retry

def is_pdf(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    # Check extension and guessed MIME type
    return filename.lower().endswith(".pdf") or (t or "").endswith("pdf")

def is_image(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    image_exts = (".jpg", ".jpeg", ".png")
    # Check extension and guessed MIME type
    return filename.lower().endswith(image_exts) or (t or "").startswith("image")

def is_text(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    # Check extension and guessed MIME type
    return filename.lower().endswith(".txt") or (t or "") == "text/plain"

def is_docx(filename):
    if not filename: return False
    t, _ = guess_type(filename)
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    # Check extension and guessed MIME type
    return filename.lower().endswith(".docx") or (t or "") == docx_mime

# Add retry mechanism to Vision calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception))
def extract_text_from_image_via_vision(image_file, openai_api_key=None):
    """Extracts text from an image using OpenAI Vision API, with retries."""
    client = OpenAI(
        api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=120.0 # Longer timeout for vision potentially
    )
    logger.debug(f"Requesting Vision text extraction for image: {image_file}")
    try:
        with open(image_file, "rb") as f:
            data = f.read()
            mime_type = guess_type(image_file)[0] or "image/png" # Default to png if guess fails
            b64 = base64.b64encode(data).decode("utf-8")
            image_url = f"data:{mime_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "auto" # Use 'auto' or 'high' based on needs
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract all the computer-readable text from this image as accurately as possible. Avoid commentary, return only the extracted text."
                    },
                ]
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-mini", # Ensure this model supports vision
            messages=messages,
            max_tokens=32768,
            temperature=0,
        )
        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"Vision extraction successful for {image_file}. Text length: {len(extracted_text)}")
        return extracted_text
    except Exception as e:
        logger.error(f"Vision extraction failed for {image_file}. Error: {e}")
        raise # Reraise for retry




def generate_audio(
    input_method: str,
    files: Optional[List[str]],
    input_text: Optional[str],
    language: str = "English",
    openai_api_key: str = None,
) -> (str, str, str, str): # Added 4th str for the hidden gr.File component
    """Generates audio from uploaded files or direct text input."""
    start_time = time.time()
    
    # API Key Check - one-api (at OPENAI_BASE_URL via resolved_openai_api_key) handles all TTS routing.
    # It needs its own API key (OPENAI_API_KEY or the one from UI input).
    if not (openai_api_key or os.getenv("OPENAI_API_KEY")): # Check if any source provides the key for one-api
        raise gr.Error("Mr.🆖 AI Hub API Key is required.")

    # Resolve OpenAI API key and Base URL once (used for dialogue and audio generation)
    resolved_openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    resolved_openai_base_url = os.getenv("OPENAI_BASE_URL")
    
    full_text = ""
    gr.Info("📦 Processing input...")
    podcast_title_base = "Group Discussion" # Default title base

    if input_method == "Upload Files":
        if not files:
            raise gr.Error("Please upload at least one file or switch to another input method.")
        texts = []
        file_names = []
        for file_path in files:
            if not file_path:
                logger.warning("Received an empty file path in the list, skipping.")
                continue
            actual_file_path = file_path.name if hasattr(file_path, 'name') else file_path
            file_path_obj = Path(actual_file_path)
            file_names.append(file_path_obj.stem)
            logger.info(f"Processing file: {file_path_obj.name}")
            text = ""

            if is_pdf(str(file_path_obj)):
                try:
                    with file_path_obj.open("rb") as f:
                        reader = PdfReader(f)
                        if reader.is_encrypted:
                             logger.warning(f"Skipping encrypted PDF: {file_path_obj.name}")
                             raise gr.Error(f"Cannot process password-protected PDF: {file_path_obj.name}")
                        page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                        text = "\n\n".join(page_texts) if page_texts else ""
                        if not text: logger.warning(f"No text extracted from PDF: {file_path_obj.name}")
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path_obj.name}: {e}")
                    if "PdfReadError" in str(type(e)):
                         raise gr.Error(f"Error reading PDF file: {file_path_obj.name}. It might be corrupted or improperly formatted.")
                    else:
                         raise gr.Error(f"Error processing PDF file: {file_path_obj.name}.")
            elif is_image(str(file_path_obj)):
                try:
                    text = extract_text_from_image_via_vision(str(file_path_obj), resolved_openai_api_key)
                except Exception as e:
                    logger.error(f"Error processing image {file_path_obj.name} with Vision API: {e}")
                    raise gr.Error(f"Error extracting text from image: {file_path_obj.name}. Check API key, file format, and OpenAI status. Error: {e}")
            elif is_text(str(file_path_obj)):
                try:
                    with open(actual_file_path, "r", encoding="utf-8", errors='ignore') as f: 
                        text = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path_obj.name}: {e}")
                    raise gr.Error(f"Error reading text file: {file_path_obj.name}. Check encoding. Error: {e}")
            elif is_docx(str(file_path_obj)):
                try:
                    doc = docx.Document(actual_file_path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text]
                    text = "\n\n".join(paragraphs)
                    if not text: logger.warning(f"No text extracted from DOCX: {file_path_obj.name}")
                except Exception as e:
                    logger.error(f"Error reading DOCX file {file_path_obj.name}: {e}")
                    if "PackageNotFoundError" in str(type(e)):
                        raise gr.Error(f"Error reading DOCX file: {file_path_obj.name}. It might be corrupted, not a valid DOCX format, or password-protected.")
                    else:
                        raise gr.Error(f"Error processing DOCX file: {file_path_obj.name}.")
            else:
                try:
                   f_size = file_path_obj.stat().st_size
                   if f_size > 0:
                       raise gr.Error(f"Unsupported file type: {file_path_obj.name}. Please upload TXT, PDF, DOCX, or image file (JPG, JPEG, PNG). Note: Older .doc format is not supported.")
                   else:
                       logger.warning(f"Skipping empty or placeholder file: {file_path_obj.name}")
                       text = ""
                except FileNotFoundError:
                    logger.warning(f"File not found during processing, likely a temporary file issue: {actual_file_path}")
                    text = ""
                except Exception as e:
                     logger.error(f"Error checking file status for {file_path_obj.name}: {e}")
                     raise gr.Error(f"Error accessing file: {file_path_obj.name}.")
            texts.append(text)
        full_text = "\n\n".join(filter(None, texts))
        if not full_text.strip():
             raise gr.Error("Could not extract any text from the uploaded file(s). Please check the files or try different ones.")
        if file_names:
            podcast_title_base = file_names[0] if len(file_names) == 1 else f"{len(file_names)} Files"


    elif input_method == "Enter Topic":
        if not input_text or not input_text.strip():
            raise gr.Error("Please enter topic or switch to another input method.")
        full_text = input_text
        podcast_title_base = "Pasted Topic"

    

    else:
        raise gr.Error("Invalid input method selected.")

    logger.info(f"Total input text length: {len(full_text)} characters.")
    if not full_text.strip(): # Double check after all input methods
        raise gr.Error("No text content to process. Please provide valid input.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15), retry=retry_if_exception_type(ValidationError))
    @llm(
        model="gpt-4.1-mini", # This LLM call might still use OpenAI
        api_key=resolved_openai_api_key, # Dialogue generation still uses OpenAI key
        base_url=resolved_openai_base_url,
        temperature=0.5,
        max_tokens=16384
    )
    def generate_dialogue(text: str, language: str) -> Dialogue:
        """
        You are a language tutor helping Hong Kong secondary students improve their speaking skills, particularly for group discussion or interaction.
        
        Your task is to take the input text provided and create a dialogue between 4 students who are engaged in a group discussion on the topic provided in the input text. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the discussion topic and question prompts as well as any relevant key points or interesting facts from the text accompanying the discussion topic for the group discussion.

        Important: The ENTIRE dialogue (including brainstorming, scratchpad, and actual dialogue) should be written in {language}. If 'Chinese' or 'Cantonese', use correct idiomatic Traditional Chinese (繁體中文) suitable for a Hong Kong audience.

        Here is the input text you will be working with:

        <input_text>
        {text}
        </input_text>

        First, carefully read through the input text and identify the discussion topic and question prompts, as well as any relevant key points or interesting facts from the text accompanying the discussion topic.

        <scratchpad>
        Brainstorm creative ways to discuss the main topic and the question prompts you identified in the input text.

        Keep in mind that your dialogue should model an authentic discussion and interaction among 4 students in an oral exam setting. It must demonstrate the communication skills and strategies that are expected to be found in an effective group discussion.

        During the group discussion, the students are expected to: 
        - Use strategies to initiate an interaction (e.g. Today, we are here to discuss the proposal to ... | Let's begin by talking about the reasons why ...).
        - Use strategies to maintain an interaction (e.g. What do you guys think? | Alright, does anyone have any other ideas to add? | OK, shall we move on to discuss ...).
        - Use strategies to respond to an interaction (e.g. I agree. | That's an interesting suggestion, but I'm a bit worried that ... | I'm sorry I don't agree.).
        - Use strategies to rephrase another student's ideas when needed (e.g. I see what you mean. You were saying that ...).
        - Use a range of accurate vocabulary and language patterns.
        - Express a range of well-developed ideas clearly, with elaboration and detail.

        Write your brainstorming ideas and a rough outline for the dialogue here. Be sure to note the communication strategies you want to incorporate into your dialogue.
        </scratchpad>

        Now that you have brainstormed ideas and created a rough outline, it's time to write the actual dialogue. Aim for a natural, conversational flow between the 4 speakers. Incorporate the ideas from your brainstorming session and make sure to incorporate the expected communication strategies in a deliberate manner.

        <podcast_dialogue>
        Write an engaging, informative dialogue here. Use a conversational tone and include any relevant context or explanations from the text accompanying the discussion topic. Use 'Candidate A', 'Candidate B', 'Candidate C', 'Candidate D' to identify the 4 speakers. Do not include any bracketed placeholders like [Candidate A] or [Candidate B]. Design your output to be read aloud -- it will be directly converted into audio. Assign appropriate speakers (Candidate A, Candidate B, Candidate C, Candidate D) to each line, varying them for a natural conversation with natural turn-taking. Ensure the output strictly adheres to the required format: a list of objects, each with 'text' and 'speaker' fields.

        Make the dialogue 8-10 minutes long, staying on topic and maintaining an engaging flow.

        At the end of the dialogue, have one of the speakers very briefy summarize the main ideas from their discussion in 1-2 sentences.
        </podcast_dialogue>
        """

    try:
        gr.Info("✨ Generating dialogue script with AI...")
        llm_start_time = time.time()
        llm_output = generate_dialogue(full_text, language)
        logger.info(f"Dialogue generation took {time.time() - llm_start_time:.2f} seconds.")

    except ValidationError as e:
        logger.error(f"LLM output validation failed after retries: {e}")
        raw_output = getattr(e, 'llm_output', str(e)) 
        raise gr.Error(f"The AI model returned an unexpected format even after retries. Please try again or simplify the input. Raw output hint: {str(raw_output)[:500]}...")
    except Exception as e:
        logger.error(f"Error during dialogue generation: {e}")
        error_str = str(e).lower()
        if "authentication" in error_str:
             raise gr.Error("Authentication error with API. Please check your API key.")
        elif "rate limit" in error_str:
             raise gr.Error("API rate limit exceeded. Please wait and try again, or check your usage tier.")
        elif "base_url" in error_str or "connection" in error_str: # This refers to OpenAI connection for dialogue
            dialogue_base_url = resolved_openai_base_url or 'the configured OpenAI endpoint for dialogue generation'
            raise gr.Error(f"Could not connect to {dialogue_base_url}. Please check the URL and network connection.")
        elif "invalid request" in error_str and "image" in error_str:
             raise gr.Error("Error processing image with Vision API. The image might be invalid, unsupported, or the model doesn't support image input.")
        else:
            raise gr.Error(f"An error occurred during dialogue generation: {e}")

    if not llm_output or not llm_output.dialogue:
        raise gr.Error("The AI failed to generate a dialogue script. The input might be too short or unclear.")

    characters = 0
    total_lines = len(llm_output.dialogue)
    logger.info(f"Starting TTS generation for {total_lines} dialogue lines.")
    gr.Info(f"🪄 Generating audio for {total_lines} dialogue lines... (this may take a while)")

    results = [None] * total_lines

    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_index = {}
        # The comprehensive API key check for resolved_openai_api_key (used by get_mp3 for one-api)
        # was done at the beginning of the generate_audio function.
        # If that check passed, we can proceed.
        future_to_index = {
            executor.submit(get_mp3, line.text, line.voice(language), resolved_openai_api_key, language): i # Pass UI 'language' as 'language_selection'
            for i, line in enumerate(llm_output.dialogue) if line.text.strip()
        }
        
        for line in llm_output.dialogue:
            if line.text.strip():
                characters += len(line.text)

        processed_count = 0
        tts_start_time = time.time()
        for future in cf.as_completed(future_to_index):
            index = future_to_index[future]
            line_obj = llm_output.dialogue[index]
            transcript_line = f"{line_obj.speaker}: {line_obj.text}"
            try:
                audio_chunk = future.result()
                results[index] = (transcript_line, audio_chunk)
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_lines: 
                     gr.Info(f"🪄 Generated audio for {processed_count}/{total_lines} lines...")
            except Exception as exc:
                 logger.error(f'TTS generation failed for line {index+1} after retries: {exc}')
                 error_msg = f"[TTS Error: Failed audio for line {index+1}]"
                 results[index] = (transcript_line, error_msg) 

    logger.info(f"TTS generation took {time.time() - tts_start_time:.2f} seconds.")
    logger.info(f"Total characters for TTS: {characters}")

    gr.Info("🧩 Combining audio segments...")
    final_audio_chunks = []
    final_transcript_lines = []
    successful_lines = 0
    for i, result in enumerate(results):
        line_obj = llm_output.dialogue[i] 
        if result is None:
            if line_obj.text.strip(): 
                logger.error(f"Result missing for non-empty line {i+1}. Original text: '{line_obj.text[:50]}...' Skipping.")
                final_transcript_lines.append(f"[Internal Error: Audio result missing for line {i+1}] {line_obj.speaker}: {line_obj.text}")
            continue 

        transcript_part, audio_part = result
        final_transcript_lines.append(transcript_part) 
        if isinstance(audio_part, bytes):
            final_audio_chunks.append(audio_part)
            successful_lines += 1
    
    if not final_audio_chunks:
        if any("[TTS Error" in line for line in final_transcript_lines):
             raise gr.Error("Failed to generate audio for all lines. Please check the transcript for details and review API key/status.")
        else:
             raise gr.Error("Failed to generate any audio, although dialogue script was created. Check TTS service status or API key.")

    audio = b"".join(final_audio_chunks)
    transcript = "\n\n".join(final_transcript_lines)

    logger.info(f"Successfully generated audio for {successful_lines}/{total_lines} lines.")

    temporary_directory = "./gradio_cached_files/tmp/" 
    os.makedirs(temporary_directory, exist_ok=True)

    try:
        temp_file_path = None
        with NamedTemporaryFile(
            dir=temporary_directory,
            delete=False, 
            suffix=".mp3",
            prefix="GI_audio_"
        ) as temp_file:
             temp_file.write(audio)
             temp_file_path = temp_file.name 

        if temp_file_path:
             logger.info(f"Audio saved to temporary file: {temp_file_path}")
        else:
             raise IOError("Temporary file path was not obtained.")

    except Exception as e:
        logger.error(f"Failed to write temporary audio file: {e}")
        raise gr.Error("Failed to save the generated audio file.")

    try:
        for file in glob.glob(f"{temporary_directory}GI_audio_*.mp3"):
            if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60: 
                try:
                    os.remove(file)
                    logger.debug(f"Removed old temp file: {file}")
                except OSError as e_rem:
                     logger.warning(f"Could not remove old temp file {file}: {e_rem}") 
    except Exception as e: 
        logger.warning(f"Error during old temp file cleanup: {e}")

    total_duration = time.time() - start_time
    tts_cost = (characters / 1_000_000) * 15
    if language in ["Cantonese"]:
        tts_cost *= 8
        gr.Info(f"🎉 Audio generation complete! Total time: {total_duration:.2f} seconds.")
        gr.Info(f"💸 This audio generation costs US${tts_cost:.2f}.")
    elif language in ["Chinese"]:
        tts_cost *= 2
        gr.Info(f"🎉 Audio generation complete! Total time: {total_duration:.2f} seconds.")
        gr.Info(f"💸 This audio generation costs US${tts_cost:.2f}.")
    else:
        gr.Info(f"🎉 Audio generation complete! Total time: {total_duration:.2f} seconds.")
        gr.Info(f"💸 This audio generation costs US${tts_cost:.2f}.")

    # Prepare audio title for history
    # Get current time in UTC
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    # Define Hong Kong timezone
    hk_tz = pytz.timezone('Asia/Hong_Kong')
    # Convert UTC time to Hong Kong time
    hk_now = utc_now.astimezone(hk_tz)
    # Format the Hong Kong time
    final_podcast_title = f"{podcast_title_base} - {hk_now.strftime('%Y-%m-%d %H:%M')}"
    
    # Escape transcript for JavaScript string literal
    escaped_transcript = transcript.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')

    # Create JavaScript to call the save function in head.html
    # The audio file path needs to be accessible by the client's browser.
    # Gradio serves files from a temporary location. We need to ensure this path is correct.
    # If temp_file_path is absolute, we might need to make it relative or ensure it's served.
    # For Gradio, files in `file_output` are typically served under `/file=...`
    # We assume `temp_file_path` as returned by `gr.Audio` is directly fetchable.
    
    # The audio_output component in Gradio will have a URL like /file=/path/to/temp_file.mp3
    # We need to pass this server-relative path to the JS function.
    # `temp_file_path` is the absolute path on the server.
    # Gradio's `gr.Audio(type="filepath")` returns the absolute path.
    # When this path is set as the value of an `gr.Audio` output, Gradio makes it accessible via a `/file=` URL.
    # The JS in `head.html` will fetch this URL.

    # Construct the web-accessible URL for the temporary file
    # Gradio serves files from gr.Audio(type="filepath") via /file=<path>
    # Ensure the path is properly URL encoded if it contains special characters, though Gradio might handle this.
    # For simplicity, we'll assume basic paths for now. If issues persist, URL encoding might be needed.
    # gradio_file_url = f"/file={temp_file_path}" # Original absolute path approach

    temp_file_path_obj = Path(temp_file_path)
    try:
        # Path.cwd() in the context of the uvicorn server (likely /app in Docker)
        app_root = Path.cwd()
        relative_temp_path = temp_file_path_obj.relative_to(app_root)
        gradio_file_url = f"/file={relative_temp_path}"
        logger.info(f"Constructed relative gradio_file_url: {gradio_file_url}")
    except ValueError:
        # Fallback if relative_to fails (e.g. different drives or not a subpath)
        gradio_file_url = f"/file={temp_file_path}" # Keep the absolute path
        logger.warning(f"Could not make path relative to CWD ('{app_root}'). Using absolute path for gradio_file_url: {gradio_file_url}")


    # The audio_url will now be fetched by JS from a hidden gr.File component
    # We pass the raw temp_file_path to that component.
    # The JSON will just contain a marker or the title to correlate.
    data_to_send = {
        "title": final_podcast_title,
        # "audio_url": gradio_file_url, # REMOVED - JS will get this from hidden gr.File
        "audio_file_component_id": "temp_audio_file_url_holder", # ID of the hidden gr.File
        "transcript": transcript,
        "tts_cost": f"{tts_cost:.2f}" # Added tts_cost, formatted as string
    }
    json_data_string = json.dumps(data_to_send)
    
    logger.debug(f"Returning JSON data for JS trigger (no audio_url, JS will fetch from component): {json_data_string[:200]}...")

    return temp_file_path, transcript, json_data_string, temp_file_path # 4th item for hidden gr.File


# --- Gradio UI Definition ---

allowed_extensions = [
    ".txt", ".pdf", ".docx", ".jpg", ".jpeg", ".png" 
]

examples_dir = Path("examples")
examples = [
    [ # Input method, files, text, language, api_key
        "Upload Files", [str(examples_dir / "DSE 2019 Paper 4 Set 2.2.png")], "", "English", None
    ],
    [
        "Upload Files", [str(examples_dir / "DSE 2023 Paper 4 Set 1.1.png")], "", "English", None
    ]
]

def read_file_content(filepath: str, default: str = "") -> str:
    try:
        return Path(filepath).read_text(encoding='utf-8') 
    except FileNotFoundError:
        logger.warning(f"{filepath} not found, using default content.")
        return default
    except Exception as e:
         logger.error(f"Error reading file {filepath}: {e}. Using default.")
         return default


description_md = read_file_content("description.md", "Generate a sample group discussion with accompanying audio from a provided topic.")
footer_md = read_file_content("footer.md", "")
head_html = read_file_content("head.html", "")


with gr.Blocks(theme="ocean", title="Mr.🆖 DiscussAI 🎙️🎧", css="footer{display:none !important}") as demo: # Reverted allowed_paths
    gr.Markdown(description_md)

    with gr.Row():
        input_method_radio = gr.Radio(
            ["Upload Files", "Enter Topic"],
            label="📁 Sources",
            value="Upload Files"
        )

    with gr.Group(visible=True) as file_upload_group:
        file_input = gr.Files(
            label="Upload JPG, JPEG, PNG, TXT, PDF, or DOCX Files",
            file_types=allowed_extensions,
            file_count="multiple",
        )

    with gr.Group(visible=False) as text_input_group: 
        text_input = gr.Textbox(
            label="✍️ Enter Topic",
            lines=10,
            placeholder="Paste or type your discussion topic here..."
        )
    
    

    lang_input = gr.Radio(
            label="🌐 Discussion Language",
            choices=["English", "Chinese", "Cantonese"],
            value="English",
        )

    API_KEY_URL = "https://api.mr5ai.com"
    with gr.Accordion("⚙️ Advanced Settings", open=False):
        gr.Markdown(
            f"💡 Get your Mr.🆖 AI Hub API Key [here]({API_KEY_URL})"
        )
        api_key_input = gr.Textbox(
                label="Mr.🆖 AI Hub API Key",
                type="password",
                placeholder="sk-xxx",
                elem_id="mr_ng_ai_hub_api_key_input"
        )

    submit_button = gr.Button("✨ Generate Discussion with Audio", variant="primary")

    with gr.Column():
        audio_output = gr.Audio(label="🔊 Audio", type="filepath", elem_id="podcast_audio_player") # Keep existing elem_id
        transcript_output = gr.Textbox(label="📃 Transcript", lines=15, show_copy_button=True, autoscroll=False, elem_id="podcast_transcript_display") # Keep existing elem_id

    with gr.Accordion("📜 Archives (Stored in your browser)", open=False): # Keep existing Accordion
        # This HTML component will be populated by JavaScript from head.html
        podcast_history_display = gr.HTML("<ul id='podcastHistoryList' style='list-style-type: none; padding: 0;'><li>Loading archives...</li></ul>")
        # Hidden Textbox component to pass JSON data to JavaScript
        js_trigger_data_textbox = gr.Textbox(label="JS Trigger Data", visible=False, elem_id="js_trigger_data_textbox")
        # Hidden File component to get a Gradio-served URL for the audio
        temp_audio_file_output_for_url = gr.File(label="Temp Audio File URL Holder", visible=False, elem_id="temp_audio_file_url_holder")


    def switch_input_method(choice):
        """Updates visibility and clears the inactive input fields."""
        is_upload = choice == "Upload Files"
        is_text = choice == "Enter Topic"

        # Determine visibility updates
        file_vis = is_upload
        text_vis = is_text

        # Determine value updates (clear hidden fields)
        # gr.update() means no change to value
        file_val_update = gr.update(value=None) if not is_upload else gr.update()
        text_val_update = gr.update(value="") if not is_text else gr.update()

        return {
            file_upload_group: gr.update(visible=file_vis),
            text_input_group: gr.update(visible=text_vis),
            file_input: file_val_update,
            text_input: text_val_update,
        }

    input_method_radio.change(
        fn=switch_input_method,
        inputs=input_method_radio,
        outputs=[
            file_upload_group, 
            text_input_group, 
            file_input, 
            text_input
        ]
    )

    submit_button.click(
        fn=generate_audio,
        inputs=[ # Order must match generate_audio parameters
            input_method_radio,
            file_input,
            text_input,
            lang_input,
            api_key_input
        ],
        outputs=[audio_output, transcript_output, js_trigger_data_textbox, temp_audio_file_output_for_url],
        api_name="generate_audio"
    )

    gr.Examples(
        examples=examples,
        inputs=[ # Ensure order matches generate_audio parameters for examples
            input_method_radio, 
            file_input, 
            text_input, 
            lang_input, 
            api_key_input
        ],
        # Examples won't trigger the history save directly unless we adapt the example fn or outputs
        # For now, history save is only for manual generation.
        outputs=[audio_output, transcript_output, js_trigger_data_textbox, temp_audio_file_output_for_url],
        fn=generate_audio,
        cache_examples=True,
        run_on_click=True,
        label="Examples (Click for Demo)"
    )

    gr.Markdown(footer_md)
    demo.head = (os.getenv("HEAD", "") or "") + head_html

# --- App Setup & Launch ---

demo = demo.queue(
    max_size=20,
    default_concurrency_limit=5, 
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    examples_dir.mkdir(exist_ok=True)
    example_files = [
        "DSE 2019 Paper 4 Set 2.2.png",
        "DSE 2023 Paper 4 Set 1.1.png"
    ]
    for fname in example_files:
        fpath = examples_dir / fname
        if not fpath.is_file():
            logger.warning(f"Example file {fpath} not found. Creating empty placeholder.")
            try:
                fpath.touch()
            except OSError as e:
                logger.error(f"Failed to create placeholder file {fpath}: {e}")

    os.makedirs("./gradio_cached_files/tmp/", exist_ok=True)

    logger.info("Starting Gradio application via Uvicorn...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
