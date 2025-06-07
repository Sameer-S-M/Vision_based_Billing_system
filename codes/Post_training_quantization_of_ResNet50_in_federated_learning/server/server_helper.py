from collections import OrderedDict


from omegaconf import DictConfig

import torch

from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):


        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader, cfg):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        device = torch.device("cpu")

   

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict})

        model.load_state_dict(state_dict, strict=True)




        loss, accuracy = test(model, testloader, device)

        if(server_round >= cfg.num_rounds): 
            save_path = "./model_weights"
            torch.save(model.state_dict(), f"{save_path}/global_model_round_{server_round}.pth")


        return loss, {"accuracy": accuracy}

    return evaluate_fn


