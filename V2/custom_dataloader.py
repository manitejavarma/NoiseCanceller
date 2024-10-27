from torch.utils.data import DataLoader

from V2.custom_dataset import AudioDataSetCustom


def create_data_loader(clean_paths, noisy_paths, batch_size, shuffle=True):
    dataset = AudioDataSetCustom(clean_paths, noisy_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)