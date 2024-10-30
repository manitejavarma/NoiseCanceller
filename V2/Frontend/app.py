from flask import Flask, request, render_template, send_file
from pydub import AudioSegment
from io import BytesIO
import os

app = Flask(__name__)


# Route for main page
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling audio upload
@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return "No file uploaded", 400

    file = request.files['audio']
    audio = AudioSegment.from_file(file)  # Load the uploaded audio file

    # Process the audio file (example: increase volume by 6dB)
    processed_audio = audio + 6  # Increase volume

    # Save processed audio to a BytesIO buffer
    buffer = BytesIO()
    processed_audio.export(buffer, format="wav")
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="processed_audio.wav", mimetype="audio/wav")


if __name__ == '__main__':
    app.run(debug=True)