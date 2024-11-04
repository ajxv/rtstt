import sounddevice as sd
import numpy as np
import threading
import queue
import whisper
import time
import logging
from typing import Callable, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class TranscriptionService:
    def __init__(self, callback_fn: Callable[[str], None]):
        """
        Initialize the transcription service.
        
        Args:
            callback_fn: Function to call with transcribed text.
        """
        # Audio parameters
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 1  # in seconds
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.ENERGY_THRESHOLD = 0.002  # Adjust for ambient noise

        # State management
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callback_fn = callback_fn
        
        # Initialize the Whisper model (medium size)
        self.model = whisper.load_model("medium")
        logging.info("Whisper model loaded successfully.")

    def audio_callback(self, indata, frames, time, status):
        """Handles incoming audio data in chunks and adds them to the queue."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.flatten().astype(np.float32))

    def transcribe_audio(self):
        """Processes queued audio chunks and transcribes them using Whisper."""
        last_transcription_time = 0
        min_gap = 0.75  # Minimum gap in seconds between transcriptions

        while self.is_recording:
            if not self.audio_queue.empty():
                # Collect all available audio chunks from the queue
                audio_chunk = np.concatenate(list(self.audio_queue.queue), axis=0).astype(np.float32)
                self.audio_queue.queue.clear()  # Clear queue after collecting chunks
                
                # Check audio volume and timing for transcription
                if (np.mean(np.abs(audio_chunk)) > self.ENERGY_THRESHOLD and 
                    (time.time() - last_transcription_time) > min_gap):
                    
                    last_transcription_time = time.time()
                    result = self.model.transcribe(audio_chunk, language='en', without_timestamps=True)
                    transcribed_text = result['text'].strip()

                    if transcribed_text:
                        logging.info(f"Transcribed Text: {transcribed_text}")
                        self.callback_fn(transcribed_text)

    def start(self):
        """Starts the audio stream and transcription thread."""
        if not self.is_recording:
            self.is_recording = True
            threading.Thread(target=self.transcribe_audio, daemon=True).start()
            
            self.stream = sd.InputStream(
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                callback=self.audio_callback
            )
            self.stream.start()
            logging.info("Transcription service started.")
            return True
        logging.warning("Transcription service is already running.")
        return False

    def stop(self):
        """Stops the audio stream and transcription."""
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                logging.info("Audio stream stopped.")
            logging.info("Transcription service stopped.")
            return True
        logging.warning("Transcription service is not running.")
        return False

    def __del__(self):
        """Ensures resources are cleaned up if the service is deleted."""
        self.stop()
        logging.info("Transcription service cleaned up.")