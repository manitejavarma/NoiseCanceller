import pickle

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from V2.autoencoder import DenoisingAutoencoder
from V2.custom_dataloader import create_data_loader
from V2.custom_dataset import AudioDataSetCustom
from V2.utilities import setup_device, get_clean_noise_paths
from sklearn.model_selection import train_test_split

def train_step(model, data_loader, loss_fn, optimizer, epoch):
    model.train()
    train_loss = 0
    for clean_speech, noisy_speech in data_loader:
        clean_speech, noisy_speech = clean_speech.to(device), noisy_speech.to(device)
        optimizer.zero_grad()
        output = model(noisy_speech)
        loss = loss_fn(output, clean_speech)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn):
    model.eval()
    loss = 0
    with torch.inference_mode():
        for clean_speech, noisy_speech in data_loader:
            clean_speech, noisy_speech = clean_speech.to(device), noisy_speech.to(device)
            output = model(noisy_speech)
            loss += loss_fn(output, clean_speech).item()
    return loss / len(data_loader)


# Model Persistence
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


# Main Training Loop
def train(model, train_loader, test_loader, optimizer, criterion, epochs):
    writer = SummaryWriter()
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, criterion, optimizer, epoch)
        test_loss = eval_model(model, test_loader, criterion)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


def load_hyperparameters():
    return {
        "frame_size": 1024,
        "hop_length": 512,
        "duration": 7,  # in seconds
        "sample_rate": 16000,
        "mono": True,
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.001
    }


if __name__ == "__main__":
    device = setup_device()
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
    train_dataset = create_data_loader(clean_train, noisy_train, hyperparameters['batch_size'])
    test_dataset = create_data_loader(clean_test, noisy_test, hyperparameters['batch_size'], shuffle=False)

    model = DenoisingAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = nn.MSELoss()

    train(model, train_dataset, test_dataset, optimizer, criterion, hyperparameters['epochs'])
