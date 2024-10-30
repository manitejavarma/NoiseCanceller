import pickle

import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from custom_dataloader import create_data_loader
from model_manager import ModelManager
from models.unet import UNet
from utilities import setup_device, get_clean_noise_paths
from sklearn.model_selection import train_test_split
from constants import *


def train_step(model, data_loader, loss_fn, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for clean_speech, noisy_speech in tqdm(data_loader, desc="Epoch {}".format(epoch + 1)):
        clean_speech, noisy_speech = clean_speech.to(device), noisy_speech.to(device)
        optimizer.zero_grad()
        output = model(noisy_speech)
        loss = loss_fn(output, clean_speech)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    loss = 0
    with torch.inference_mode():
        for clean_speech, noisy_speech in data_loader:
            clean_speech, noisy_speech = clean_speech.to(device), noisy_speech.to(device)
            output = model(noisy_speech)
            loss += loss_fn(output, clean_speech).item()
    return loss / len(data_loader)


# Main Training Loop
def train(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    writer = SummaryWriter()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = train_step(model, train_loader, criterion, optimizer, epoch, device)
        test_loss = eval_model(model, test_loader, criterion, device)
        writer.add_scalar(f"{model.__class__.__name__}/Loss/train", train_loss, epoch)
        writer.add_scalar(f"{model.__class__.__name__}/Loss/test", test_loss, epoch)
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    device = setup_device()
    min_max_values = None
    with open(MODEL_MIN_MAX_PATH, "rb") as file:
        min_max_values = pickle.load(file)

    clean_files, noisy_files = get_clean_noise_paths(MODEL_INPUT_DIR)

    clean_train, clean_test, noisy_train, noisy_test = train_test_split(clean_files, noisy_files, test_size=0.2,
                                                                        random_state=69)

    train_dataset = create_data_loader(clean_train, noisy_train, BATCH_SIZE, shuffle=True)
    test_dataset = create_data_loader(clean_test, noisy_test, BATCH_SIZE, shuffle=False)  # No need to shuffle test data

    # Model Initialization
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train(model, train_dataset, test_dataset, optimizer, criterion, EPOCHS, device)

    # Save the model
    model_save_name = CURRENT_MODEL_NAME
    modelManager = ModelManager(model, model_save_name)
    modelManager.save()
