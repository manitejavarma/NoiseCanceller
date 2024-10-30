import torch
import os

class ModelManager:
    def __init__(self, model, model_name, base_dir="models/"):
        self.model = model
        self.model_name = model.__class__.__name__ if model_name is None else model_name
        # check if base_dir exits
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.path = base_dir + self.model_name + ".pth"

    def save(self):
        torch.save(self.model.state_dict(), self.path)
        print(f"Model saved to {self.path}")

    def load(self):
        self.model.load_state_dict(torch.load(self.path))
        print(f"Model loaded from {self.path}")