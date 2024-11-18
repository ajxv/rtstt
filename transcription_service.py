# transcription_service.py
import numpy as np
import threading
import queue
import whisper
import time
import logging
from typing import Callable, Optional

class TranscriptionService:
    def __init__(self, callback_fn: Callable[[str], None]):
        """
        Initialize the transcription service.
        
        Args:
            callback_fn: Function to call with transcribed text.
        """
        # Audio parameters
        self.SAMPLE_RATE = 16000
        self.ENERGY_THRESHOLD = 0.002  # Adjust for ambient noise

        # State management
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callback_fn = callback_fn
        self.processing_thread = None

        # Initialize the Whisper model
        # Available models: tiny, base, small, medium, large
        # - tiny: Fastest but least accurate
        # - base: A balance between speed and accuracy
        # - small: More accurate, slower than base
        # - medium: Even more accurate, slower than small
        # - large: Most accurate but slowest
        # https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
        
        self.model = whisper.load_model("medium")
        logging.info("Whisper model loaded successfully.")

    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Process incoming audio chunk from the client.
        
        Args:
            audio_chunk (np.ndarray): Audio data as numpy array
        """
        if self.is_recording:
            self.audio_queue.put(audio_chunk)

    def process_queue(self):
        """Processes queued audio chunks and transcribes them using Whisper."""
        last_transcription_time = 0
        min_gap = 0.75  # Minimum gap in seconds between transcriptions
        
        while self.is_recording:
            if not self.audio_queue.empty():
                # Collect all available audio chunks
                chunks = []
                while not self.audio_queue.empty():
                    chunks.append(self.audio_queue.get())
                
                audio_data = np.concatenate(chunks)

                # Check audio volume and timing for transcription
                if (np.mean(np.abs(audio_data)) > self.ENERGY_THRESHOLD and
                    (time.time() - last_transcription_time) > min_gap):
                    
                    last_transcription_time = time.time()
                    result = self.model.transcribe(audio_data, language='en', without_timestamps=True)
                    transcribed_text = result['text'].strip()
                    
                    if transcribed_text:
                        logging.info(f"Transcribed Text: {transcribed_text}")
                        self.callback_fn(transcribed_text)
            
            time.sleep(0.1)  # Prevent busy-waiting

    def start(self):
        """Starts the transcription processing thread."""
        if not self.is_recording:
            self.is_recording = True
            self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
            self.processing_thread.start()
            logging.info("Transcription service started.")
            return True
        
        logging.warning("Transcription service is already running.")
        return False

    def stop(self):
        """Stops the transcription processing."""
        if self.is_recording:
            self.is_recording = False
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)
            self.audio_queue.queue.clear()
            logging.info("Transcription service stopped.")
            return True
        
        logging.warning("Transcription service is not running.")
        return False

    def __del__(self):
        """Ensures resources are cleaned up if the service is deleted."""
        self.stop()
        logging.info("Transcription service cleaned up.")