# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
from transcription_service import TranscriptionService
import logging
import base64
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize Flask app and Socket.IO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow external connections

def emit_transcription(text: str):
    """Emit transcribed text to the connected client via Socket.IO."""
    socketio.emit('transcription', {'text': text})
    logging.info("Emitted transcription to client.")

# Initialize TranscriptionService with the emit callback
transcription_service = TranscriptionService(callback_fn=emit_transcription)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data from the client."""
    if transcription_service.is_recording:
        audio_bytes = base64.b64decode(data['audio_chunk'])
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        transcription_service.process_audio_chunk(audio_array)

@socketio.on('start_recording')
def handle_start_recording():
    """Start audio recording and transcription."""
    success = transcription_service.start()
    message = 'Recording started' if success else 'Already recording'
    logging.info(message)
    return {'status': 'success' if success else 'error', 'message': message}

@socketio.on('stop_recording')
def handle_stop_recording():
    """Stop audio recording and transcription."""
    success = transcription_service.stop()
    message = 'Recording stopped' if success else 'Not recording'
    logging.info(message)
    return {'status': 'success' if success else 'error', 'message': message}

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection and stop transcription."""
    transcription_service.stop()
    logging.info("Client disconnected. Transcription stopped.")

if __name__ == '__main__':
    port = 5000
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
