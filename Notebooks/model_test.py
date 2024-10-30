import librosa
import soundfile as sf
import numpy as np
import os
import torch
import pickle
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import glob

from timeit import default_timer as timer
from tqdm import tqdm

import torch.optim as optim

import torch
import torch.nn as nn


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train

def load_spectrogram(spectrogram_path):
    spectrogram = np.load(spectrogram_path, allow_pickle=True)
    spectrogram = torch.tensor(spectrogram, dtype = torch.float32)
    return spectrogram

class AudioDataSetCustom(Dataset):
    def __init__(self, clean_speech_paths, noisy_speech_paths):
        self.clean_speech_paths = clean_speech_paths
        self.noisy_speech_paths = noisy_speech_paths

    def __len__(self):
        return len(self.clean_speech_paths)

    def __getitem__(self, index):
        clean_speech, noisy_speech = load_spectrogram(self.clean_speech_paths[index]), load_spectrogram(self.noisy_speech_paths[index])
        clean_speech = clean_speech.unsqueeze(0)
        noisy_speech = noisy_speech.unsqueeze(0)
        return clean_speech, noisy_speech


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    train_loss = 0
    model.to(device)
    for data in data_loader:
        clean_speech, noisy_speech = data
        clean_speech, noisy_speech = clean_speech.to(device), noisy_speech.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(noisy_speech)

        # Compute the loss
        loss = loss_fn(output, clean_speech)

        train_loss += loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} ")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    loss = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)

        print("Device is ", next(model.parameters()).device)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item()}

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clean_dir = '../input/spectrograms/clean'
    noisy_dir = '../input/spectrograms/noise'
    clean_files = glob.glob(os.path.join(clean_dir + '/' + '*.npy'))
    noisy_files = glob.glob(os.path.join(noisy_dir + '/' + '*.npy'))
    clean_files = [clean_file.replace('\\', '/') for clean_file in clean_files]
    noisy_files = [noisy_file.replace('\\', '/') for noisy_file in noisy_files]

    clean_train, clean_test, noisy_train, noisy_test = train_test_split(
        clean_files, noisy_files, test_size=0.2, random_state=42)

    train_dataset = AudioDataSetCustom(clean_train, noisy_train)
    test_dataset = AudioDataSetCustom(clean_test, noisy_test)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=128,  # how many samples per batch?
                                  num_workers=0,  # how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True)  # shuffle the data?

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=128,
                                 num_workers=0,
                                 shuffle=False)  # don't usually need to shuffle testing data

    model = DenoisingAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_time_start_on_gpu = timer()

    epochs = 500
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
                   model=model,
                   loss_fn=criterion,
                   optimizer=optimizer
                   )
        test_step(data_loader=test_dataloader,
                  model=model,
                  loss_fn=criterion
                  )

    train_time_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                                end=train_time_end_on_gpu,
                                                device=device)

    #listen to signals
    ipd.Audio(clean_speech[0].numpy(), rate=16000)