import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import librosa
import subprocess
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from threading import Lock
import time
import mimetypes
import logging
from autocorrect import Speller
from collections import Counter
import langdetect
import requests

# GPU Diagnostics
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("No GPU detected. Falling back to CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spell checker
spell = Speller(lang='en')

# Initialize models
print("Initializing speech-to-text model...")
asr_model = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(asr_model)
speech_model = WhisperForConditionalGeneration.from_pretrained(asr_model).to(device).eval()
print(f"Speech-to-text model loaded on {device}. GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

print("Initializing summarization model...")
model_name = "facebook/bart-large-cnn"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    print(f"Summarization model loaded on {device}. GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
except Exception as e:
    logger.error(f"Error loading summarization model: {e}")
    raise

file_lock = Lock()

# Supported languages for the dropdown
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Swedish": "sv",
    "Detect Automatically": "auto"
}

# Audio Preprocessing
def preprocess_audio(audio_data, sample_rate=16000):
    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    if len(audio_data) < sample_rate:
        audio_data = np.pad(audio_data, (0, sample_rate - len(audio_data)))
    return audio_data

def convert_to_wav(audio_path):
    output_wav = "temp_audio.wav"
    print(f"Converting {audio_path} to WAV format...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_wav, "-y"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
        print("Audio conversion completed.")
        return output_wav
    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}")

# Speech-to-Text with Language Support
def process_audio_segment(segment, sr=16000, language="en"):
    input_features = processor(segment, sampling_rate=sr, return_tensors="pt").input_features
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = speech_model.generate(
            input_features,
            forced_bos_token_id=processor.get_decoder_prompt_ids(language=language)[0][1] if language != "auto" else None
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Segment processed. GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    return transcription.lower() if transcription is not None else ""

# Translation Function Using API Key
def translate_to_english(text, source_language="auto"):
    if not text or source_language == "en":
        return text  # Skip translation for English or empty text
    try:
        if source_language == "auto":
            detected = langdetect.detect(text)
            if detected == "en":
                return text
            source_language = detected

        # Google Translate API key
        API_KEY = "YOUR_API_KEY_HERE"  # INSERT YOUR API KEY HERE (e.g., "AIzaSy...")
        MAX_CHARACTERS = 100000
        translated_chunks = []

        # Split text into sentences to preserve context
        sentences = text.split(". ")
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_length + len(sentence) + 2 <= MAX_CHARACTERS:
                current_chunk += sentence + ". "
                current_length += len(sentence) + 2
            else:
                if current_chunk:
                    url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"
                    payload = {
                        "q": current_chunk.strip(),
                        "source": source_language,
                        "target": "en",
                        "format": "text"
                    }
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        translated_text = response.json()["data"]["translations"][0]["translatedText"].lower()
                        translated_chunks.append(translated_text)
                    else:
                        logger.error(f"Translation API error: {response.text}")
                        translated_chunks.append(current_chunk.strip())  # Fallback
                current_chunk = sentence + ". "
                current_length = len(sentence) + 2

        # Translate the final chunk
        if current_chunk:
            url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"
            payload = {
                "q": current_chunk.strip(),
                "source": source_language,
                "target": "en",
                "format": "text"
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                translated_text = response.json()["data"]["translations"][0]["translatedText"].lower()
                translated_chunks.append(translated_text)
            else:
                logger.error(f"Translation API error: {response.text}")
                translated_chunks.append(current_chunk.strip())

        translated_text = " ".join(translated_chunks)
        logger.info(f"Translated from {source_language} to English: {translated_text[:100]}...")
        return translated_text

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # Fallback to original text

# Summarization Functions
def correct_typos_contextually(input_text):
    try:
        words = input_text.split()
        word_freq = Counter(words)
        corrected_words = []
        for word in words:
            if word_freq[word] > 2 or len(word) < 3:
                corrected_words.append(word)
                continue
            corrected_word = spell(word)
            corrected_words.append(corrected_word)
        corrected_text = " ".join(corrected_words)
        logger.info(f"Corrected text: {corrected_text}")
        return corrected_text
    except Exception as e:
        logger.error(f"Error in typo correction: {e}")
        return input_text

def summarize_text(input_text, base_max_length=200, base_min_length=50):
    try:
        if not input_text or input_text.strip() == "":
            return "no valid text provided"
        
        corrected_text = correct_typos_contextually(input_text)
        input_length = len(corrected_text.split())
        logger.info(f"Input length: {input_length} words")

        tokens = tokenizer.encode(corrected_text, truncation=False)
        max_input_tokens = 1000
        summaries = []
        if len(tokens) > max_input_tokens:
            words = corrected_text.split()
            chunk_size = max_input_tokens // 2
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            logger.info(f"Text split into {len(chunks)} chunks")
        else:
            chunks = [corrected_text]

        for chunk in chunks:
            chunk_length = len(chunk.split())
            if chunk_length < 20:
                summaries.append(chunk)
                continue
            elif chunk_length < 100:
                max_length = max(50, int(chunk_length * 1.2))
                min_length = max(20, int(chunk_length * 0.4))
            elif chunk_length < 300:
                max_length = base_max_length
                min_length = base_min_length
            else:
                max_length = min(400, int(chunk_length * 0.7))
                min_length = min(100, int(chunk_length * 0.25))

            inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                summary_ids = summarizer_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=1.0,
                    num_beams=6,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary.lower())
            print(f"Summary chunk generated. GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        final_summary = " ".join(summaries)
        logger.info(f"Final summary: {final_summary}")
        return final_summary
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"failed to summarize: {str(e)}"

# Main Audio Processing Function
def process_audio_to_summary(audio_path, language, progress=gr.Progress()):
    if not audio_path:
        return "no file uploaded", "", "", ""

    print(f"Received audio path: {audio_path}, Language: {language}")
    start_time = time.time()
    progress(0, desc="starting transcription...")
    
    try:
        mime_type, _ = mimetypes.guess_type(audio_path)
        print(f"Detected MIME type: {mime_type}")
        valid_mime_types = {'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/aac'}
        valid_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.aac'}
        if (mime_type not in valid_mime_types) or not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            return f"invalid file type: {audio_path} (mime: {mime_type})", "", "", ""

        with file_lock:
            if not audio_path.lower().endswith(".wav"):
                audio_path = convert_to_wav(audio_path)
                progress(0.1, desc="audio converted to wav")

        print("Loading and preprocessing audio...")
        speech, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(speech) / sr
        speech = preprocess_audio(speech)
        progress(0.2, desc="audio loaded and preprocessed")

        chunk_length = 10 * sr
        chunks = [speech[i:i + chunk_length] for i in range(0, len(speech), chunk_length)]
        print(f"Audio split into {len(chunks)} chunks.")
        progress(0.3, desc=f"split into {len(chunks)} chunks")

        language_code = LANGUAGE_OPTIONS.get(language, "auto")
        transcriptions = []
        transcribe_start = time.time()
        for i, chunk in enumerate(chunks, 1):
            transcription = process_audio_segment(chunk, language=language_code)
            transcriptions.append(transcription)
            print(f"Processed chunk {i}/{len(chunks)}. GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            progress((0.3 + 0.4 * i / len(chunks)), desc=f"transcribing chunk {i}/{len(chunks)}")

        final_transcription = " ".join(transcriptions).strip()
        transcribe_time = time.time() - transcribe_start
        progress(0.7, desc="transcription complete")

        # Translation
        translation_start = time.time()
        translated_text = translate_to_english(final_transcription, language_code)
        translation_time = time.time() - translation_start
        progress(0.85, desc="translation complete")

        # Summarization
        summary_start = time.time()
        summary = summarize_text(translated_text)
        summary_time = time.time() - summary_start
        progress(0.95, desc="summarization complete")

        with file_lock:
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
                print("Temporary file cleaned up.")

        duration = time.time() - start_time
        print(f"Processing completed in {duration:.2f} seconds!")
        progress(1.0, desc="processing complete!")

        # Metrics
        metrics = (f"Audio Duration: {audio_duration:.2f}s\n"
                   f"Time to Transcribe: {transcribe_time:.2f}s\n"
                   f"Time to Translate: {translation_time:.2f}s\n"
                   f"Time to Summarize: {summary_time:.2f}s")

        return (f"transcription (original):\n\n{final_transcription}",
                f"translated text (English):\n\n{translated_text}",
                f"summary:\n\n{summary}",
                metrics)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"error: {str(e)}", "", "", ""

# Text Processing Function
def process_text_to_summary(input_text, language):
    if not input_text:
        return "no text provided", "", ""

    language_code = LANGUAGE_OPTIONS.get(language, "auto")
    translation_start = time.time()
    translated_text = translate_to_english(input_text.lower(), language_code)
    translation_time = time.time() - translation_start

    summary_start = time.time()
    summary = summarize_text(translated_text)
    summary_time = time.time() - summary_start

    metrics = (f"Time to Translate: {translation_time:.2f}s\n"
               f"Time to Summarize: {summary_time:.2f}s")

    return f"translated text (English):\n\n{translated_text}", f"summary:\n\n{summary}", metrics

# Gradio UI
with gr.Blocks(css="""
    body { background-color: #121212; color: white; }
    .gradio-container { background-color: #1e1e1e; padding: 30px; border-radius: 15px; }
    .title { color: #ffffff; font-size: 30px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .description { color: #b0b0b0; font-size: 16px; margin-bottom: 40px; text-align: center; }
    #audio-box, #text-box, #transcript-box, #translated-box, #summary-box, #metrics-box { 
        background-color: #333333; color: white; border-radius: 8px; padding: 15px; 
        transition: all 0.3s ease; 
    }
    #audio-box:hover, #text-box:hover, #transcript-box:hover, #translated-box:hover, #summary-box:hover, #metrics-box:hover { 
        background-color: #444444; box-shadow: 0 0 15px rgba(255, 255, 255, 0.1); 
    }
    .gradio-button { transition: all 0.3s ease; }
    .gradio-button:hover { background-color: #555555; transform: scale(1.05); }
""") as interface:
    gr.Markdown("# Multilingual Audio & Text Summarization Tool")
    gr.Markdown("Upload audio or input text to transcribe/translate/summarize. Select the language or choose 'Detect Automatically'.")
    
    with gr.Tab("Audio to Summary"):
        audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio File", elem_id="audio-box")
        language_dropdown = gr.Dropdown(
            choices=list(LANGUAGE_OPTIONS.keys()),
            value="Detect Automatically",
            label="Select Audio Language",
            info="Choose the language of the audio or select 'Detect Automatically'."
        )
        audio_submit_btn = gr.Button("Summarize Audio")
        with gr.Row():
            audio_transcript_output = gr.Textbox(label="Transcription (Original)", lines=5, elem_id="transcript-box")
            audio_translated_output = gr.Textbox(label="Translated Text (English)", lines=5, elem_id="translated-box")
            audio_summary_output = gr.Textbox(label="Summary", lines=5, elem_id="summary-box")
            audio_metrics_output = gr.Textbox(label="Performance Metrics", lines=5, elem_id="metrics-box")
        
        audio_submit_btn.click(
            fn=process_audio_to_summary,
            inputs=[audio_input, language_dropdown],
            outputs=[audio_transcript_output, audio_translated_output, audio_summary_output, audio_metrics_output]
        )
    
    with gr.Tab("Text to Summary"):
        text_input = gr.Textbox(label="Enter Text to Summarize", lines=5, placeholder="Paste your text here...", elem_id="text-box")
        text_language_dropdown = gr.Dropdown(
            choices=list(LANGUAGE_OPTIONS.keys()),
            value="Detect Automatically",
            label="Select Text Language",
            info="Choose the language of the text or select 'Detect Automatically'."
        )
        text_submit_btn = gr.Button("Summarize Text")
        with gr.Row():
            text_translated_output = gr.Textbox(label="Translated Text (English)", lines=5, elem_id="translated-box")
            text_summary_output = gr.Textbox(label="Summary", lines=5, elem_id="summary-box")
            text_metrics_output = gr.Textbox(label="Performance Metrics", lines=5, elem_id="metrics-box")
        
        text_submit_btn.click(
            fn=process_text_to_summary,
            inputs=[text_input, text_language_dropdown],
            outputs=[text_translated_output, text_summary_output, text_metrics_output]
        )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860, debug=True)
    print("Interface launched successfully! Access it via the provided URL.")