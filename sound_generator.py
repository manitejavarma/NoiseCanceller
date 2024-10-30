from Preprocess.preprocess import MinMaxNormaliser
import torch
import librosa
import numpy as np


class SoundGenerator:
    def __init__(self, hop_length, griffinlim_iter=50):
        self.hop_length = hop_length
        self.frame_size = hop_length * 2
        self.griffinlim_iter = griffinlim_iter
        self.normalizer = MinMaxNormaliser(-1, 1)

    def generate(self, spectrogram, min_max_values, phase=False):
        if phase:
            return self._convert_spectrogram_to_audio_with_phase(spectrogram, min_max_values)
        return self._convert_spectrogram_to_audio(spectrogram, min_max_values)

    def _convert_spectrogram_to_audio(self, log_spectrogram, min_max_value):
        denorm_spec = self.normalizer.denormalise(
            log_spectrogram, torch.tensor(min_max_value["min"], dtype=torch.float32),
            torch.tensor(min_max_value["max"], dtype=torch.float32)
        )
        spec = librosa.db_to_amplitude(denorm_spec.numpy())

        spec = np.pad(spec, ((0, 1), (0, 0)), mode='constant')

        return librosa.griffinlim(spec, n_iter=self.griffinlim_iter, hop_length=self.hop_length,
                                  win_length=self.frame_size, n_fft=self.frame_size)

    def _convert_spectrogram_to_audio_with_phase(self, log_spectrogram, min_max_value):
        denorm_spec, denorm_phase = self.normalizer.denormalise(
            log_spectrogram[0, :, :],
            torch.tensor(min_max_value["min"], dtype=torch.float32),
            torch.tensor(min_max_value["max"], dtype=torch.float32),
            log_spectrogram[1, :, :]
        )
        magnitude = librosa.db_to_amplitude(denorm_spec.numpy())
        complex_spectrogram = magnitude * np.exp(1j * denorm_phase.numpy())
        return librosa.istft(complex_spectrogram, hop_length=self.hop_length, win_length=self.frame_size)
