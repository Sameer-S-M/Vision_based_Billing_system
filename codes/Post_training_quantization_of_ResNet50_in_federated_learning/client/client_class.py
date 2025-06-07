from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl

from model import Net, train, test 

class FlowerClient(fl.client.NumPyClient):
    """Flower Client that uses a ResNet50 model."""

    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes)

       
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def set_parameters(self, parameters: NDArrays):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v, dtype=self.model.state_dict()[k].dtype) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar] = None) -> NDArrays:
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Train the model received by the server using local data."""
        self.set_parameters(parameters)

        lr = config.get("lr", 0.001)  
        momentum = config.get("momentum", 0.9)
        epochs = config.get("local_epochs", 1)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on the local validation data."""
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine."""
    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
        )
    return client_fn


