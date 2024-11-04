# transcription_service.py
import sounddevice as sd
import numpy as np
import threading
import queue
import whisper
import time
from typing import Callable, Optional

class TranscriptionService:
    def __init__(self, callback_fn: Callable[[str], None]):
        """
        Initialize the transcription service.
        
        Args:
            callback_fn: Function to call with transcribed text
        """
        # Audio parameters
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 1
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.ENERGY_THRESHOLD = 0.002
        
        # State management
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread: Optional[threading.Thread] = None
        self.stream: Optional[sd.InputStream] = None
        self.callback_fn = callback_fn
        
        # Initialize Whisper model
        self.model = whisper.load_model("small")

    def audio_callback(self, indata, frames, time, status):
        """Handle incoming audio data."""
        if status:
            print(f"Status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy().flatten().astype(np.float32))

    def transcribe_audio(self):
        """Process audio from queue and transcribe using Whisper."""
        last_transcription_time = 0
        min_gap = 0.75

        while self.is_recording:
            if not self.audio_queue.empty():
                audio_data = []
                while not self.audio_queue.empty():
                    audio_data.append(self.audio_queue.get())
                
                audio_chunk = np.concatenate(audio_data, axis=0).astype(np.float32)

                if (np.mean(np.abs(audio_chunk)) > self.ENERGY_THRESHOLD and 
                    (time.time() - last_transcription_time) > min_gap):
                    
                    last_transcription_time = time.time()
                    result = self.model.transcribe(
                        audio_chunk, 
                        language='en', 
                        without_timestamps=True
                    )
                    transcribed_text = result['text'].strip()
                    
                    if transcribed_text:
                        self.callback_fn(transcribed_text)

    def start(self):
        """Start the transcription service."""
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(
                target=self.transcribe_audio, 
                daemon=True
            )
            self.recording_thread.start()
            
            self.stream = sd.InputStream(
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                callback=self.audio_callback
            )
            self.stream.start()
            return True
        return False

    def stop(self):
        """Stop the transcription service."""
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            if self.recording_thread:
                self.recording_thread.join()
            return True
        return False

    def __del__(self):
        """Cleanup resources."""
        if self.is_recording:
            self.stop()