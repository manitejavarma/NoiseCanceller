import pickle

from sklearn.model_selection import train_test_split
from torch import optim, nn

from V2.custom_dataloader import create_data_loader
from V2.sound_generator import SoundGenerator
from V2.train import load_hyperparameters, eval_model
from V2.unet import UNet
from V2.utilities import setup_device, get_clean_noise_paths, load_spectrogram, get_signal_from_spectrogram
from model_manager import ModelManager
import soundfile as sf





if __name__ == '__main__':
    device = setup_device()
    model = UNet().to(device)
    ModelManager(model, "unet").load()

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
    sample_number = 59
    # sf.write('clean_test_speech.wav',
    #          get_signal(clean_test[sample_number], min_max_values, hyperparameters['hop_length']),
    #          hyperparameters['sample_rate'])
    # sf.write('noisy_test_speech.wav',
    #          get_signal(noisy_test[sample_number], min_max_values, hyperparameters['hop_length']),
    #          hyperparameters['sample_rate'])

    #with model output
    input = load_spectrogram(noisy_test[sample_number]).unsqueeze(0).unsqueeze(0).to(device)
    output = model(input)

    model_input_signals = get_signal_from_spectrogram(input.squeeze(0).squeeze(0).detach().cpu(),
                                                      noisy_test[sample_number], min_max_values,
                                                      hyperparameters['hop_length'])
    model_output_signals = get_signal_from_spectrogram(output.squeeze(0).squeeze(0).detach().cpu(),
                                                       clean_test[sample_number],
                                                       min_max_values, hyperparameters['hop_length'])
    sf.write('clean_test_speech.wav',
             model_output_signals,
             hyperparameters['sample_rate'])
    sf.write('noisy_test_speech.wav', model_input_signals,
             hyperparameters['sample_rate'])


