import librosa
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm

sampling_rate = 16000
min_audio_length = 7  # seconds
noise_folder = 'musan.tar/musan/noise/free-sound/'
clean_speech_folder = 'train-clean-100.tar/train-clean-100/LibriSpeech/train-clean-100/'
noisy_speech_folder_extracted = 'dataset/noisy_speech/'
clean_speech_folder_extracted = 'dataset/clean_speech/'


def load_audio(file_path, sr=16000):
    if file_path.endswith('.flac'):
        audio, _ = sf.read(file_path)
        return audio
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def match_length(signal, target_length):
    if len(signal) >= target_length:
        return signal[:target_length]
    repeat_count = (target_length // len(signal)) + 1
    return np.tile(signal, repeat_count)[:target_length]


def add_noise_to_speech(speech, noise, snr_db):
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 ** (snr_db / 10)
    noise_scaling_factor = np.sqrt(speech_power / (snr * noise_power))
    noisy_speech = speech + noise_scaling_factor * noise
    return noisy_speech


def get_all_noise_files():
    return [file for file in os.listdir(noise_folder) if file.endswith('.wav')]


def get_clean_speech_files():
    clean_speech_files_list = []
    for root, _, files in os.walk(clean_speech_folder):
        for file in files:
            if file.endswith('.flac'):
                clean_speech_files_list.append(os.path.join(root, file))
    return clean_speech_files_list


def pick_next_noise_file(noise_file_number):
    noise_files = get_all_noise_files()
    picked_noise_file = noise_files[noise_file_number % len(noise_files)]
    return load_audio(os.path.join(noise_folder, picked_noise_file))


def add_random_noise_to_speech_files(clean_speech_file, file_number, noise_file_number, min_length, snr_db=10):
    clean_speech = load_audio(clean_speech_file)
    picked_noise = pick_next_noise_file(noise_file_number)
    noise_matched = match_length(picked_noise, len(clean_speech))
    snr_db = np.random.randint(7, 15) #TODO: Remove reference to snr_db
    noisy_speech = add_noise_to_speech(clean_speech, noise_matched, snr_db)

    noisy_output_file = os.path.join(noisy_speech_folder_extracted, f'noisy_speech_{file_number}.wav')
    sf.write(noisy_output_file, noisy_speech[:min_length * sampling_rate], sampling_rate)

    clean_output_file = os.path.join(clean_speech_folder_extracted, f'clean_speech_{file_number}.wav')
    sf.write(clean_output_file, clean_speech[:min_length * sampling_rate], sampling_rate)


def find_min_length(clean_speech_files):
    min_length = np.inf
    for file in tqdm(clean_speech_files, desc='Finding min length'):
        audio = load_audio(file)
        min_length = min(min_length, len(audio) / sampling_rate)
    return min_length  # 1.75 seconds for libre speech


def filter_files(clean_speech_files, min_length):
    file_number = 1
    noise_file_number = 1
    filtered_files_list = []
    for file in tqdm(clean_speech_files, desc='Adding noise to files of minimum speech length'):
        audio = load_audio(file)
        if len(audio) / sampling_rate >= min_length:
            filtered_files_list.append(file)
            add_random_noise_to_speech_files(file, file_number, noise_file_number, min_length, snr_db=10)
            file_number += 1
            noise_file_number += 1
            if file_number == 2069: #TODO: Remove hardcoding
                break
    return filtered_files_list


if __name__ == "__main__":
    clean_speech_files = get_clean_speech_files()
    filtered_files = filter_files(clean_speech_files, min_audio_length)
    print(len(filtered_files))
