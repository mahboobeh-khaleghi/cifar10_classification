import numpy as np
import pandas as pd
from tqdm import tqdm 

import torch
import torch.nn as nn

import utils
from nets import MyModel
from dataloaders import Cifar10DataLoader, Normalizer, Standardizer


def eval(
    config,
    model = None,
    eval_loader = None,
    criterion = None,
):
    
    ################ Device ################
    # training device
    device = "cuda" if config.device == "gpu" and torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    
    
    if eval_loader == None:
        ################ Dataloader ################
        # Evaluation dataset
        eval_dataset = Cifar10DataLoader(
            pickles_path=config.pickles_path_eval,
            transform = None
        )
        
        # Evaluation Data Transformation
        config.preprocessing_method = config.preprocessing_method.lower()
        
        if config.preprocessing_method == "normalized":
            transform = Normalizer()
        elif config.preprocessing_method == "standardized":
            mean = torch.mean(eval_dataset.data, axis=0)
            std = torch.std(eval_dataset.data, axis=0)
            transform = Standardizer(mean = mean, std = std)
        elif config.preprocessing_method == "none":
            transform = None
        else:
            print("Error!! Undefinded transformation.")
            exit()

        # Setting the transform
        eval_dataset.transform = transform
        
        ################ Torch Dataloader ################
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size = config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=False
        )
    
   
    ################ Model ################
    # instantiation a model
    if model == None: 
        model = MyModel(config.model, vectorize_input=config.vectorize_input).to(dtype=torch.float64, device=device)
        
        # Load pretrained parameters
        model.load_state_dict(torch.load(config.pretrained_model))  
    
    
    ################ Loss Function ################
    if criterion == None: 
        config.loss_type = config.loss_type.lower()
        
        if config.loss_type == "softmax_cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif config.loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            print("Error!! Undefinded loss function.")
            exit()

    
    ################ Reporting ################
    # constructing a dataframe for evaluating report
    eval_report = pd.DataFrame({
        "batch_index": [],
        "batch_size": [],
        "accuracy": [],
        "ongoing_accuracy": [],
        "loss": [],
    })
    

    ################ Meters ################
    # Constructing an average meter for calculating the ongoing average
    accuracy_meter = utils.AverageMeter()


    ################ Evaluation ################
    with torch.no_grad():
        # Set to eval mode to change behavior of Dropout, BatchNorm
        model.eval()  
        
        with(tqdm(eval_loader)) as t_eval_loader:
            
            t_eval_loader.set_description(f"Evaluation") 
            
            for ix, (data, labels) in enumerate(t_eval_loader):
                
                # migrate current batch data and labels to the selected device
                data = data.to(device)
                labels = labels.to(device)
                
                # forward propagation
                outputs = model(data).to(torch.float64)
                
                # calculating the loss
                loss = criterion(outputs, labels)
                
                # Calculating accuracy
                accuracy = utils.accuracy(outputs, labels).cpu().item()
                
                # updating the average meter
                accuracy_meter.update(values=[accuracy])
        
                # current batch report
                batch_report = pd.DataFrame({
                        "batch_index": [ix],
                        "batch_size": [data.shape[0]],
                        "accuracy": [accuracy],
                        "ongoing_accuracy": [accuracy_meter.avg],
                        "loss": [loss.item()],
                    })

                # concatination of batch report and total report
                eval_report = pd.concat([eval_report, batch_report],ignore_index=True)
        
                # Setting tqdm postfix
                t_eval_loader.set_postfix({
                    "loss": loss.item(),
                    "accuracy": accuracy_meter.avg,
                    "device": device
                })
                
    return eval_report