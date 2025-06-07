import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from dataset import prepare_dataset
from server_helper import get_on_fit_config, get_evaluate_fn
# import torch
import os

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="../conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare  dataset

    testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 4. Define  strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  
        min_fit_clients=cfg.num_clients_per_round_fit, 
        fraction_evaluate=0.0, 
        min_evaluate_clients=cfg.num_clients_per_round_eval,  
        min_available_clients=cfg.num_clients, 
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ), 
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, cfg),
    ) 

    ## 5. Start Simulation


    history=fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )




if __name__ == "__main__":
    main()