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

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_device():
    """Determine the best available device for computation"""
    return "cpu"  # Forcing CPU for reliability

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transcribe audio files using Whisper-small model')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input audio file')
    parser.add_argument('--output', '-o', type=str, default='.',
                       help='Path to output directory (defaults to current directory)')
    parser.add_argument('--chunk-length', type=int, default=30,
                       help='Length of audio chunks in seconds (default: 30)')
    parser.add_argument('--chunkless', '-cls', action='store_true',
                       help='Process the entire audio file as one chunk')
    return parser.parse_args()

def load_audio(audio_path):
    """Load and preprocess audio file"""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logging.info(f"Loading audio file: {audio_path}")
        
        # Get audio file information
        audio_info = sf.info(audio_path)
        logging.info(f"Audio file details: {audio_info}")
        
        # Load audio with soundfile first to check if it's valid
        audio_sf, sr_orig = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio_sf.shape) > 1:
            audio_sf = audio_sf.mean(axis=1)
        
        # Resample to 16kHz using librosa
        audio = librosa.resample(audio_sf, orig_sr=sr_orig, target_sr=16000)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        duration = len(audio)/16000
        logging.info(f"Audio duration: {duration:.2f} seconds")
        logging.info(f"Audio shape: {audio.shape}")
        
        return audio, 16000
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise

def split_audio(audio: np.ndarray, sample_rate: int, chunk_length: int) -> List[Tuple[np.ndarray, float, float]]:
    """Split audio into chunks with timestamps"""
    chunk_length_samples = chunk_length * sample_rate
    chunks = []
    
    for i in range(0, len(audio), chunk_length_samples):
        chunk = audio[i:i + chunk_length_samples]
        if len(chunk) < sample_rate:  # Skip chunks shorter than 1 second
            continue
            
        # Pad last chunk if needed
        if len(chunk) < chunk_length_samples:
            chunk = np.pad(chunk, (0, chunk_length_samples - len(chunk)))
            
        start_time = i / sample_rate
        end_time = (i + len(chunk)) / sample_rate
        chunks.append((chunk, start_time, end_time))
    
    return chunks

def transcribe_chunk(chunk: np.ndarray, model, processor) -> str:
    """Transcribe a single audio chunk"""
    input_features = processor(
        chunk, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(model.device)

    predicted_ids = model.generate(
        input_features,
        language="en",
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
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_audio_file(audio_path: str, use_chunks: bool = True, chunk_length: int = 30):
    """Transcribe audio file with or without chunking"""
    try:
        # Load model and processor
        logging.info("Loading Whisper model and processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(get_device())
        
        # Load and preprocess audio
        audio, sr = load_audio(audio_path)
        
        if use_chunks:
            # Process in chunks
            logging.info(f"Processing audio in {chunk_length}-second chunks...")
            chunks = split_audio(audio, sr, chunk_length)
            transcription = ""
            total_chunks = len(chunks)
            
            for i, (chunk, start_time, end_time) in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{total_chunks}")
                chunk_text = transcribe_chunk(chunk, model, processor)
                start_stamp = format_timestamp(start_time)
                end_stamp = format_timestamp(end_time)
                transcription += f"[{start_stamp} --> {end_stamp}] {chunk_text}\n"
        else:
            # Process entire file at once
            logging.info("Processing entire audio file as one chunk...")
            transcription = transcribe_chunk(audio, model, processor)
        
        return transcription.strip()

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def save_transcription(transcription: str, output_path: Path):
    """Save transcription to text file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info(f"Transcription saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving transcription: {e}")
        raise

def main():
    """Main function to run the transcription pipeline"""
    # Set up logging
    setup_logging()
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare output file path
        input_filename = Path(args.input).stem
        output_path = output_dir / f"{input_filename}_transcription.txt"
        
        # Perform transcription
        transcription = transcribe_audio_file(
            args.input,
            use_chunks=not args.chunkless,
            chunk_length=args.chunk_length
        )
        
        # Save result
        save_transcription(transcription, output_path)
        
        logging.info("Transcription completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()