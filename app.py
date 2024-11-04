# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import sounddevice as sd
import numpy as np
import threading
import queue
import whisper
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Load Whisper model
model = whisper.load_model("small")

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
ENERGY_THRESHOLD = 0.002

# Set up the audio queue
audio_queue = queue.Queue()
is_recording = False
recording_thread = None

def audio_callback(indata, frames, time, status):
    """Callback function to handle incoming audio data."""
    if status:
        print(f"Status: {status}")
    if is_recording:
        audio_queue.put(indata.copy().flatten().astype(np.float32))

def transcribe_audio():
    """Process audio from queue and transcribe using Whisper."""
    last_transcription_time = 0
    min_gap = 0.75

    while is_recording:
        if not audio_queue.empty():
            audio_data = []
            while not audio_queue.empty():
                audio_data.append(audio_queue.get())
            
            audio_chunk = np.concatenate(audio_data, axis=0).astype(np.float32)

            if (np.mean(np.abs(audio_chunk)) > ENERGY_THRESHOLD and 
                (time.time() - last_transcription_time) > min_gap):
                
                last_transcription_time = time.time()
                result = model.transcribe(audio_chunk, language='en', without_timestamps=True)
                transcribed_text = result['text'].strip()
                
                if transcribed_text:
                    # Emit the transcription to connected clients
                    socketio.emit('transcription', {'text': transcribed_text})

def start_recording():
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        recording_thread = threading.Thread(target=transcribe_audio, daemon=True)
        recording_thread.start()
        
        # Start the audio stream
        sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=audio_callback
        ).start()

def stop_recording():
    global is_recording
    is_recording = False
    if recording_thread:
        recording_thread.join()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def handle_start_recording():
    start_recording()
    return {'status': 'success', 'message': 'Recording started'}

@socketio.on('stop_recording')
def handle_stop_recording():
    stop_recording()
    return {'status': 'success', 'message': 'Recording stopped'}

if __name__ == '__main__':
    socketio.run(app, debug=True)