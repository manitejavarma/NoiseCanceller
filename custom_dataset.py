from torch.utils.data import Dataset

from utilities import load_spectrogram


class AudioDataSetCustom(Dataset):
    def __init__(self, clean_speech_paths, noisy_speech_paths):
        self.clean_speech_paths = clean_speech_paths
        self.noisy_speech_paths = noisy_speech_paths

    def __len__(self):
        return len(self.clean_speech_paths)

    def __getitem__(self, index):
        clean_speech = load_spectrogram(self.clean_speech_paths[index])
        noisy_speech = load_spectrogram(self.noisy_speech_paths[index])
        return clean_speech, noisy_speech

