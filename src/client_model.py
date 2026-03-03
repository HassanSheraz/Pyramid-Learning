"""
client_model.py
---------------
Simple CNN for CIFAR-10 and the local training function used by every FL client.

Architecture:
    Conv(3->32, 3x3) -> ReLU -> MaxPool(2x2)
    Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
    FC(64*6*6 -> 512) -> ReLU -> Dropout(0.5)
    FC(512 -> 10)

This is intentionally lightweight so all 100-round experiments complete on CPU
in under 30 minutes on a modern laptop.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Lightweight CNN for CIFAR-10 (3x32x32 -> 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # -> 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                         # -> 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int = 10) -> SimpleCNN:
    """Construct and return a fresh SimpleCNN."""
    return SimpleCNN(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(
    global_model: nn.Module,
    data_indices: list,
    dataset,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> dict:
    """
    Run local SGD on a client's data subset and return the gradient update.

    Args:
        global_model : current global model (will NOT be modified in place)
        data_indices : list of dataset indices belonging to this client
        dataset      : full training dataset (torchvision CIFAR10)
        local_epochs : number of local SGD epochs
        batch_size   : mini-batch size
        lr           : SGD learning rate
        momentum     : SGD momentum
        weight_decay : L2 regularisation
        device       : "cpu" or "cuda"

    Returns:
        {
            "delta_weights": OrderedDict of (global_param - local_param) per layer,
            "loss"         : average cross-entropy loss over final epoch,
            "num_samples"  : number of local training samples,
        }
    """
    if len(data_indices) == 0:
        return {"delta_weights": None, "loss": 0.0, "num_samples": 0}

    # Deep-copy so we don't modify the global model
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    optimizer = optim.SGD(
        local_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    subset = Subset(dataset, data_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    # Compute delta = local_weights - global_weights  (what we send to server)
    delta_weights = {}
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()
    for key in global_state:
        delta_weights[key] = local_state[key].cpu() - global_state[key].cpu()

    return {
        "delta_weights": delta_weights,
        "loss": avg_loss,
        "num_samples": len(data_indices),
    }
