"""
MIT License

Copyright (c) 2020 Valerio Velardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


"""
1- load a file
2- extracting log spectrogram from signal
3- normalise spectrogram
4- save the normalised spectrogram

PreprocessingPipeline
"""
import os
import pickle

import librosa
import numpy as np
from constants import *


class Loader:
    """Loader is responsible for loading an audio file."""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal


class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        phase = np.angle(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram, phase


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array, phase):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        phase = (phase + np.pi) / (2 * np.pi)
        phase = phase * (self.max - self.min) + self.min
        return norm_array, phase

    def denormalise(self, norm_array, original_min, original_max, norm_phase):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        phase = (norm_phase - self.min) / (self.max - self.min)
        phase = (2 * phase - 1) * np.pi
        return array, phase


class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)

        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 '../' + MODEL_MIN_MAX_PATH)
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "ab") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        if not os.path.exists(os.path.dirname(self.feature_save_dir)):
            os.makedirs(os.path.dirname(self.feature_save_dir))

        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir + file_name[:-3] + ".npy")
        return save_path


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- extracting log spectrogram from signal
        3- normalise spectrogram
        4- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")

    def save_min_max_values(self):
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        feature, phase = self.extractor.extract(signal)
        norm_feature, norm_phase = self.normaliser.normalise(feature, phase)
        stacked_feature = np.stack((norm_feature, norm_phase), axis=0)
        save_path = self.saver.save_feature(stacked_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def process_audio(self, audio):
        signal = self.loader.load(audio)
        feature, phase = self.extractor.extract(signal)
        norm_feature, norm_phase = self.normaliser.normalise(feature, phase)
        stacked_feature = np.stack((norm_feature, norm_phase), axis=0)
        return stacked_feature, feature.min(), feature.max()

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == "__main__":

    # instantiate all objects
    preprocessing_pipeline = PreprocessingPipeline()

    preprocessing_pipeline.loader = Loader(SAMPLE_RATE, DURATION, MONO)
    preprocessing_pipeline.extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    preprocessing_pipeline.normaliser = MinMaxNormaliser(MIN_MAX_NORMALIZER_MIN, MIN_MAX_NORMALIZER_MAX)
    preprocessing_pipeline.saver = Saver(CLEAN_SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    # Preprocess for clean and noisy speech files in the dataset
    preprocessing_pipeline.process(CLEAN_SPEECH_DIR)
    preprocessing_pipeline.saver = Saver(NOISY_SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    preprocessing_pipeline.process(NOISY_SPEECH_DIR)
    preprocessing_pipeline.save_min_max_values()
