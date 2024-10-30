from constants import *
from models.unet import UNet
from Preprocess.preprocess import PreprocessingPipeline, LogSpectrogramExtractor, MinMaxNormaliser, Loader
from utilities import *
from model_manager import ModelManager
import soundfile as sf

import torch


def clean_audio(input_audio, model, device):

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = Loader(SAMPLE_RATE, DURATION, MONO)
    preprocessing_pipeline.extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    preprocessing_pipeline.normaliser = MinMaxNormaliser(-1, 1)
    stacked_feature, feature_min, feature_max = preprocessing_pipeline.process_audio(input_audio)
    # convert stacked_feature to tensor
    stacked_feature = torch.tensor(stacked_feature).to(device)

    input = stacked_feature.unsqueeze(0).to(device)
    output_spect = None
    with torch.no_grad():
        output_spect = model(input)
    clean_signal = get_signal_from_spectrogram_min_max(output_spect.squeeze(0).detach().cpu(),
                                                       {"min": feature_min, "max": feature_max},
                                                       HOP_LENGTH, phase=True)
    return clean_signal

if __name__ == '__main__':
    device = setup_device()
    model = UNet().to(device)

    # Load the model state dict
    ModelManager(model, CURRENT_MODEL_NAME).load()

    # Denoising the audio
    denoised_audio = clean_audio('untitled.wav', model, device)

    sf.write('test_output.wav', denoised_audio,
             SAMPLE_RATE)

"""
if __name__ == '__main__':
    device = setup_device()
    model = UNet().to(device)
    ModelManager(model, "UNetPhase").load()

    hyperparameters = load_hyperparameters()
    input_dir = 'input/spectrograms/'
    min_max_path = 'input/min_max_values.pkl'
    min_max_values = None
    with open(min_max_path, "rb") as file:
        min_max_values = pickle.load(file)

    clean_files, noisy_files = get_clean_noise_paths(input_dir)

    clean_train, clean_test, noisy_train, noisy_test = train_test_split(
        clean_files, noisy_files, test_size=0.2, random_state=42)

    # Run Training
    train_dataloader = create_data_loader(clean_train, noisy_train, hyperparameters['batch_size'])
    test_dataloader = create_data_loader(clean_test, noisy_test, hyperparameters['batch_size'], shuffle=False)

    # optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    # criterion = nn.MSELoss()

    # evaluate model and print loss
    # print(eval_model(model, test_dataloader, criterion, device))

    # get one sample spect -> send it to model -> get output -> use min max and save the spect as audio file
    sample_number = 69
    # sf.write('clean_test_speech.wav',
    #          get_signal(clean_test[sample_number], min_max_values, hyperparameters['hop_length']),
    #          hyperparameters['sample_rate'])
    # sf.write('noisy_test_speech.wav',
    #          get_signal(noisy_test[sample_number], min_max_values, hyperparameters['hop_length']),
    #          hyperparameters['sample_rate'])

    # #with model output
    # input = load_spectrogram(noisy_test[sample_number]).unsqueeze(0).unsqueeze(0).to(device)
    # output = model(input)
    #
    # model_input_signals = get_signal_from_spectrogram(input.squeeze(0).squeeze(0).detach().cpu(),
    #                                                   noisy_test[sample_number], min_max_values,
    #                                                   hyperparameters['hop_length'])
    # model_output_signals = get_signal_from_spectrogram(output.squeeze(0).squeeze(0).detach().cpu(),
    #                                                    clean_test[sample_number],
    #                                                    min_max_values, hyperparameters['hop_length'])
    # sf.write('clean_test_speech.wav',
    #          model_output_signals,
    #          hyperparameters['sample_rate'])
    # sf.write('noisy_test_speech.wav', model_input_signals,
    #          hyperparameters['sample_rate'])

    # After phase correction
    # get the clean signal and noisy signal from the model
    input = load_spectrogram(noisy_test[sample_number]).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input)

    model_input_signals = get_signal_from_spectrogram(input.squeeze(0).detach().cpu(),
                                                      noisy_test[sample_number], min_max_values,
                                                      hyperparameters['hop_length'], phase=True)

    model_output_signals = get_signal_from_spectrogram(output.squeeze(0).detach().cpu(),
                                                       clean_test[sample_number], min_max_values,
                                                       hyperparameters['hop_length'], phase=True)

    sf.write('clean_test_speech.wav',
             model_output_signals,
             hyperparameters['sample_rate'])
    sf.write('noisy_test_speech.wav', model_input_signals,
             hyperparameters['sample_rate'])

"""