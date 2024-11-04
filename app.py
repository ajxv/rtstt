from flask import Flask, render_template
from flask_socketio import SocketIO
from transcription_service import TranscriptionService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Initialize Flask app and Socket.IO
app = Flask(__name__)
socketio = SocketIO(app)

def emit_transcription(text: str):
    """
    Emit transcribed text to the connected client via Socket.IO.
    
    Args:
        text (str): Transcribed text to emit.
    """
    socketio.emit('transcription', {'text': text})
    logging.info("Emitted transcription to client.")

# Initialize TranscriptionService with the emit callback
transcription_service = TranscriptionService(callback_fn=emit_transcription)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@socketio.on('start_recording')
def handle_start_recording():
    """
    Start audio recording and transcription.
    
    Returns:
        dict: Status and message indicating recording state.
    """
    success = transcription_service.start()
    message = 'Recording started' if success else 'Already recording'
    logging.info(message)
    return {'status': 'success' if success else 'error', 'message': message}

@socketio.on('stop_recording')
def handle_stop_recording():
    """
    Stop audio recording and transcription.
    
    Returns:
        dict: Status and message indicating recording state.
    """
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
    # Run the app with Socket.IO support
    socketio.run(app, debug=True)
    logging.info("Server started.")
