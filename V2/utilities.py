import glob
import os

import numpy as np
import torch


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
