import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import nets
import utils
import runners
from nets import MyModel
from dataloaders import Cifar10DataLoader, Normalizer, Standardizer

def train(
    config
):
    ################ Device ################
    # training device
    device = "cuda" if config.device == "gpu" and torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # if config == "gpu" and torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    
    
    
    ################ Dataloader ################
    # Training dataset
    train_dataset = Cifar10DataLoader(
        pickles_path=config.pickles_path_train,
        transform = None
    )
    
    # Validation dataset
    validation_dataset = Cifar10DataLoader(
        pickles_path=config.pickles_path_eval,
        transform = None
    )
    
    # Training and Validation Data Transformation
    config.preprocessing_method = config.preprocessing_method.lower()
    
    if config.preprocessing_method == "normalized":
        transform = Normalizer()
    elif config.preprocessing_method == "standardized":
        mean = torch.mean(train_dataset.data, axis=0)
        std = torch.std(train_dataset.data, axis=0)
        transform = Standardizer(mean = mean, std = std)
    elif config.preprocessing_method == "none":
        transform = None
    else:
        print("Error!! Undefinded transformation.")
        exit()

    # Setting the transform
    train_dataset.transform = transform
    validation_dataset.transform = transform
    

    ################ Torch Dataloader ################
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    
    ################ Model ################
    # instantiation a model
    model = MyModel(config.model, vectorize_input=config.vectorize_input).to(dtype=torch.float64, device=device)
    
    
    
    ################ Loss Function ################
    config.loss_type = config.loss_type.lower()
    
    if config.loss_type == "softmax_cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif config.loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        print("Error!! Undefinded loss function.")
        exit()
    
    
    ################ Optimizer ################
    config.optimizer_type = config.optimizer_type.lower()
    
    if config.optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            momentum=config.momentum
        )
    elif config.optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.lr
        )
    else:
        print("Error!! Undefinded optimizer.")
        exit()
    
    
    ################ Reporting ################
    # constructing a dataframe for training report
    report = pd.DataFrame({
        "epoch": [],
        "batch_index": [],
        "batch_size": [],
        "accuracy": [],
        "ongoing_accuracy": [],
        "loss": [],
        "phase": []
    })
    

    ################ Meters ################
    # Constructing an average meter for calculating the ongoing average
    accuracy_meter = utils.AverageMeter()


    ################ Training ################
    model.train()
    # training for some epochs
    for epoch in range(config.num_epochs):  
        
        # reseting accuracy meter
        accuracy_meter.reset()
        
        # t_train_loader = tqdm(train_loader)
        with(tqdm(train_loader)) as t_train_loader:
            
            t_train_loader.set_description(f"Training @ epoch={epoch}")
            
            for ix, (data, labels) in enumerate(t_train_loader):
                
                # migrate current batch data and labels to the selected device
                data = data.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward propagation
                outputs = model(data)
                
                # calculating the loss
                loss = criterion(outputs, labels)
                
                # backward propagation
                loss.backward()
                
                # updating the model's weights
                optimizer.step()
                
                # Calculating accuracy
                accuracy = utils.accuracy(outputs, labels).cpu().item()
                
                # updating the average meter
                accuracy_meter.update(values=[accuracy])
                
                # current batch report
                batch_report = pd.DataFrame({
                        "epoch": [epoch],
                        "batch_index": [ix],
                        "batch_size": [data.shape[0]],
                        "accuracy": [accuracy],
                        "ongoing_accuracy": [accuracy_meter.avg],
                        "loss": [loss.item()],
                        "phase": ["train"]
                    })
                
                # concatination of batch report and total report
                report = pd.concat([report, batch_report],ignore_index=True)
                
                # Setting tqdm postfix
                t_train_loader.set_postfix({
                    "loss": loss.item(),
                    "accuracy": accuracy_meter.avg,
                    "device": device
                })
                
            ################ Evaluation on Validation Dataset ################
            validation_report = runners.eval(
                config = config,
                model = model,
                eval_loader = validation_loader,
                criterion = criterion
            )
            
            # Adding a phase column to validation report
            validation_report["phase"] = "val"
            
            # Adding an epoch column to validation report
            validation_report["epoch"] = epoch
            
            # Concatinating vlidation report with total report
            report = pd.concat([report, validation_report],ignore_index=True)
            
            
    torch.save(model.state_dict(), config.trained_model_saving_path)        
    return model, report