# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
from transcription_service import TranscriptionService

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize transcription service with socket.io emit callback
def emit_transcription(text: str):
    socketio.emit('transcription', {'text': text})

transcription_service = TranscriptionService(callback_fn=emit_transcription)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recording')
def handle_start_recording():
    success = transcription_service.start()
    return {
        'status': 'success' if success else 'error',
        'message': 'Recording started' if success else 'Already recording'
    }

@socketio.on('stop_recording')
def handle_stop_recording():
    success = transcription_service.stop()
    return {
        'status': 'success' if success else 'error',
        'message': 'Recording stopped' if success else 'Not recording'
    }

@socketio.on('disconnect')
def handle_disconnect():
    transcription_service.stop()

if __name__ == '__main__':
    socketio.run(app, debug=True)