import glob
import os

import numpy as np
import torch

from sound_generator import SoundGenerator


def load_spectrogram(path):
    spectrogram = np.load(path, allow_pickle=True)
    return torch.tensor(spectrogram, dtype=torch.float32)


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def get_clean_noise_paths(base_dir):
    clean_dir = base_dir + 'clean'
    noisy_dir = base_dir + 'noise'
    clean_files = glob.glob(os.path.join(clean_dir + '/' + '*.npy'))
    noisy_files = glob.glob(os.path.join(noisy_dir + '/' + '*.npy'))

    clean_files = [clean_file.replace('\\', '/') for clean_file in clean_files]
    noisy_files = [noisy_file.replace('\\', '/') for noisy_file in noisy_files]

    return clean_files, noisy_files


def get_signal(spect_path, min_max_values, hop_length):
    spect = load_spectrogram(spect_path)
    min_max_value = min_max_values[spect_path]
    return SoundGenerator(hop_length).generate(spect, min_max_value)


def get_signal_from_spectrogram(spectrogram, spect_path, min_max_values, hop_length, phase=False):
    min_max_value = min_max_values[spect_path]
    return SoundGenerator(hop_length).generate(spectrogram, min_max_value, phase)


def get_signal_from_spectrogram_min_max(spectrogram, min_max_value, hop_length, phase=False):
    return SoundGenerator(hop_length).generate(spectrogram, min_max_value, phase)
