from flask import Flask, request, render_template, send_file
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
import io

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    # Load audio data with pydub and handle any format
    audio_file = request.files['audio']
    audio = AudioSegment.from_file(audio_file).set_channels(1)  # Convert to mono

    # Set the target sample rate
    target_sample_rate = 16000
    audio = audio.set_frame_rate(target_sample_rate)

    # Convert the AudioSegment to a numpy array and normalize
    audio_data = np.array(audio.get_array_of_samples())
    if audio_data.dtype != np.int16:
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)  # Normalize and convert to int16

    # Save to a BytesIO buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, target_sample_rate, audio_data)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="audio/wav",
        as_attachment=True,
        download_name="processed_audio.wav"
    )


if __name__ == '__main__':
    app.run(debug=True)
