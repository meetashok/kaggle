import torch

checkpoint = torch.load(os.path.join(path, file))
model.load_state_dict(checkpoint["model_state_dict"])