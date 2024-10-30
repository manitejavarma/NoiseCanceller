import librosa
import numpy as np
from pesq import pesq
import torch

from Preprocess.preprocess import PreprocessingPipeline, LogSpectrogramExtractor, MinMaxNormaliser
from models.unet import UNet
from sound_generator import SoundGenerator
from utilities import setup_device
from constants import *

def calculate_snr(clean, noisy):
    # Calculate noise by subtracting clean from noisy
    noise = noisy - clean
    # Signal power
    signal_power = np.mean(clean ** 2)
    # Noise power
    noise_power = np.mean(noise ** 2)
    # SNR calculation in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# create a function to calculate the SNR of the signal before and after sending to the model
def eval_model(model):
    # Load the clean and noisy signals
    clean_signal = librosa.load('dataset/clean_speech/clean_speech_69.wav',
                                sr=16000,
                                duration=7,
                                mono=True)[0]
    noisy_signal = librosa.load('dataset/noisy_speech/noisy_speech_69.wav',
                                sr=16000,
                                duration=7,
                                mono=True)[0]
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    preprocessing_pipeline.normaliser = MinMaxNormaliser(MIN_MAX_NORMALIZER_MIN, MIN_MAX_NORMALIZER_MAX)
    # clean_feature = preprocessing_pipeline.extractor.extract(clean_signal)[:, :-3]
    # clean_min = clean_feature.min()
    # clean_max = clean_feature.max()
    # clean_norm_feature = preprocessing_pipeline.normaliser.normalise(clean_feature)

    noisy_feature = preprocessing_pipeline.extractor.extract(noisy_signal)
    noisy_min = noisy_feature.min()
    noisy_max = noisy_feature.max()
    noisy_norm_feature = preprocessing_pipeline.normaliser.normalise(noisy_feature)
    noisy_norm_feature = torch.tensor(noisy_norm_feature, dtype=torch.float32)
    input = noisy_norm_feature.unsqueeze(0).unsqueeze(0).to(device)
    output_spect = None
    with torch.no_grad():
        output_spect = model(input)

    denoised_signal = SoundGenerator(512).generate(output_spect.squeeze(0).squeeze(0).detach().cpu(), {"min": noisy_min, "max": noisy_max})

    # SNR before denoising
    snr_before = calculate_snr(clean_signal, noisy_signal)
    pesq_score = pesq(SAMPLE_RATE, clean_signal, noisy_signal, 'wb' if SAMPLE_RATE == 16000 else 'nb')
    print(f"PESQ Score before denoising: {pesq_score}")
    print(f"SNR before denoising: {snr_before} dB")

    # SNR after denoising
    snr_after = calculate_snr(clean_signal[:len(denoised_signal)], denoised_signal)
    pesq_score = pesq(SAMPLE_RATE, clean_signal, denoised_signal, 'wb' if SAMPLE_RATE == 16000 else 'nb')
    print(f"PESQ Score after denoising: {pesq_score}")
    print(f"SNR after denoising: {snr_after} dB")

    # Improvement in SNR
    snr_improvement = snr_after - snr_before
    print(f"SNR improvement: {snr_improvement} dB")


if __name__ == '__main__':
    device = setup_device()
    model = UNet().to(device)
    eval_model(model)

