#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import logging
import numpy as np
import soundfile as sf
from typing import List, Tuple
from tqdm import tqdm
import textwrap

LANGUAGE_CODES = {
    "english": "en", "en": "en",
    "chinese": "zh", "zh": "zh",
    "german": "de", "de": "de",
    "spanish": "es", "es": "es",
    "russian": "ru", "ru": "ru",
    "korean": "ko", "ko": "ko",
    "french": "fr", "fr": "fr",
    "japanese": "ja", "ja": "ja",
    "portuguese": "pt", "pt": "pt",
    "turkish": "tr", "tr": "tr",
    "polish": "pl", "pl": "pl",
    "catalan": "ca", "ca": "ca",
    "dutch": "nl", "nl": "nl",
    "arabic": "ar", "ar": "ar",
    "swedish": "sv", "sv": "sv",
    "italian": "it", "it": "it",
    "indonesian": "id", "id": "id",
    "hindi": "hi", "hi": "hi",
    "finnish": "fi", "fi": "fi",
    "vietnamese": "vi", "vi": "vi",
    "hebrew": "he", "he": "he",
    "ukrainian": "uk", "uk": "uk",
    "greek": "el", "el": "el",
    "malay": "ms", "ms": "ms",
    "czech": "cs", "cs": "cs",
    "romanian": "ro", "ro": "ro",
    "danish": "da", "da": "da",
    "hungarian": "hu", "hu": "hu",
    "tamil": "ta", "ta": "ta",
    "norwegian": "no", "no": "no",
    "thai": "th", "th": "th",
    "urdu": "ur", "ur": "ur",
    "croatian": "hr", "hr": "hr",
    "bulgarian": "bg", "bg": "bg",
    "lithuanian": "lt", "lt": "lt",
    "latin": "la", "la": "la",
    "maori": "mi", "mi": "mi",
    "malayalam": "ml", "ml": "ml",
    "welsh": "cy", "cy": "cy",
    "slovak": "sk", "sk": "sk",
    "telugu": "te", "te": "te",
    "persian": "fa", "fa": "fa",
    "latvian": "lv", "lv": "lv",
    "bengali": "bn", "bn": "bn",
    "serbian": "sr", "sr": "sr",
    "azerbaijani": "az", "az": "az",
    "slovenian": "sl", "sl": "sl",
    "kannada": "kn", "kn": "kn",
    "estonian": "et", "et": "et",
    "macedonian": "mk", "mk": "mk",
    "breton": "br", "br": "br",
    "basque": "eu", "eu": "eu",
    "icelandic": "is", "is": "is",
    "armenian": "hy", "hy": "hy",
    "nepali": "ne", "ne": "ne",
    "mongolian": "mn", "mn": "mn",
    "bosnian": "bs", "bs": "bs",
    "kazakh": "kk", "kk": "kk",
    "albanian": "sq", "sq": "sq",
    "swahili": "sw", "sw": "sw",
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_device():
    return "cpu"

def get_formatted_language_list():
    unique_langs = sorted(set(lang for lang in LANGUAGE_CODES.keys() if len(lang) > 2))
    lang_columns = textwrap.fill(", ".join(unique_langs), width=70, initial_indent="  ", subsequent_indent="  ")
    return f"Available languages:\n{lang_columns}"

def validate_language(lang: str) -> str:
    if not lang:
        return None
        
    lang = lang.lower()
    if lang in LANGUAGE_CODES:
        return LANGUAGE_CODES[lang]
    else:
        valid_languages = sorted(set(code for code in LANGUAGE_CODES.keys() if len(code) > 2))
        raise ValueError(f"Unsupported language: {lang}\nSupported languages: {', '.join(valid_languages)}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Transcribe audio files using Whisper-small model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  whisper -i audio.mp3                             # Basic usage with defaults
  whisper -i audio.mp3 -o ~/transcripts            # Specify output directory
  whisper -i audio.mp3 -l french                   # Transcribe French audio
  whisper -i audio.mp3 --chunk-length 45           # Custom chunk length
  whisper -i audio.mp3 -cln 45                     # Same as above
  whisper -i audio.mp3 --chunkless                 # Process as single chunk
  whisper -i audio.mp3 -cls                        # Same as above
  whisper --default-language spanish               # Set Spanish as default

{get_formatted_language_list()}

Note: You can use either the full language name or its code (e.g., 'french' or 'fr')""")
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input audio file')
    parser.add_argument('--output', '-o', type=str, default='.',
                       help='Path to output directory (defaults to current directory)')
    parser.add_argument('--chunk-length', '-cln', type=int, default=30,
                       help='Length of audio chunks in seconds (default: 30)')
    parser.add_argument('--chunkless', '-cls', action='store_true',
                       help='Process the entire audio file as one chunk')
    parser.add_argument('--language', '-l', type=str,
                       help='Language of the audio (e.g., english, french, spanish)')
    parser.add_argument('--default-language', type=str,
                       help='Set default language for future transcriptions')
    
    args = parser.parse_args()
    
    try:
        default_lang_file = os.path.expanduser('~/.whisper_default_language')
        
        if args.default_language:
            lang_code = validate_language(args.default_language)
            if lang_code:
                with open(default_lang_file, 'w') as f:
                    f.write(lang_code)
                logging.info(f"Default language set to: {args.default_language} ({lang_code})")
            return None
            
        if args.language:
            args.language = validate_language(args.language)
        elif os.path.exists(default_lang_file):
            with open(default_lang_file, 'r') as f:
                args.language = f.read().strip()
                logging.info(f"Using default language: {args.language}")
        else:
            args.language = "en"
            logging.info("No language specified, using English")
            
    except ValueError as e:
        parser.error(str(e))
        
    return args

def load_audio(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logging.info(f"Loading audio file: {audio_path}")
        
        audio_info = sf.info(audio_path)
        logging.info(f"Audio file details: {audio_info}")
        
        audio_sf, sr_orig = sf.read(audio_path)
        
        if len(audio_sf.shape) > 1:
            audio_sf = audio_sf.mean(axis=1)
        
        audio = librosa.resample(audio_sf, orig_sr=sr_orig, target_sr=16000)
        
        audio = audio / np.max(np.abs(audio))
        
        duration = len(audio)/16000
        logging.info(f"Audio duration: {duration:.2f} seconds")
        logging.info(f"Audio shape: {audio.shape}")
        
        return audio, 16000
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise

def split_audio(audio: np.ndarray, sample_rate: int, chunk_length: int) -> List[Tuple[np.ndarray, float, float]]:
    chunk_length_samples = chunk_length * sample_rate
    chunks = []
    
    for i in range(0, len(audio), chunk_length_samples):
        chunk = audio[i:i + chunk_length_samples]
        if len(chunk) < sample_rate:
            continue
            
        if len(chunk) < chunk_length_samples:
            chunk = np.pad(chunk, (0, chunk_length_samples - len(chunk)))
            
        start_time = i / sample_rate
        end_time = (i + len(chunk)) / sample_rate
        chunks.append((chunk, start_time, end_time))
    
    return chunks

def transcribe_chunk(chunk: np.ndarray, model, processor, language: str) -> str:
    input_features = processor(
        chunk, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(model.device)

    predicted_ids = model.generate(
        input_features,
        language=language,
        task="transcribe",
        num_beams=5,
        max_new_tokens=225
    )
    
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0].strip()
    
    return transcription

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_audio_file(audio_path: str, use_chunks: bool = True, chunk_length: int = 30, language: str = "en"):
    try:
        logging.info("Loading Whisper model and processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(get_device())
        
        audio, sr = load_audio(audio_path)
        
        if use_chunks:
            logging.info(f"Processing audio in {chunk_length}-second chunks...")
            chunks = split_audio(audio, sr, chunk_length)
            transcription = ""
            
            with tqdm(total=len(chunks), desc="Transcribing chunks") as pbar:
                for chunk, start_time, end_time in chunks:
                    chunk_text = transcribe_chunk(chunk, model, processor, language)
                    start_stamp = format_timestamp(start_time)
                    end_stamp = format_timestamp(end_time)
                    transcription += f"[{start_stamp} --> {end_stamp}] {chunk_text}\n"
                    pbar.update(1)
        else:
            logging.info("Processing entire audio file as one chunk...")
            transcription = transcribe_chunk(audio, model, processor, language)
        
        return transcription.strip()

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def save_transcription(transcription: str, output_path: Path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info(f"Transcription saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving transcription: {e}")
        raise

def main():
    setup_logging()
    
    args = parse_arguments()
    
    if args is None:
        return
        
    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_filename = Path(args.input).stem
        output_path = output_dir / f"{input_filename}_transcription.txt"
        
        transcription = transcribe_audio_file(
            args.input,
            use_chunks=not args.chunkless,
            chunk_length=args.chunk_length,
            language=args.language
        )
        
        save_transcription(transcription, output_path)
        
        logging.info("Transcription completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()