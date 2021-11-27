import torch
import torchvision
from Torch_CV_Utils.utils import data_handling, train, test, gradcam, helpers, augmentation
from Torch_CV_Utils.models import resnet
from torch.optim.lr_scheduler import StepLR, ExponentialLR, OneCycleLR, LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim

# standard_lr = 0.01
# momentum_val = 0.9
# L2_penalty = 1e-5
# sch_reduceLR_mode = "min"
# sch_reduceLR_factor = 0.1
# sch_reduceLR_patience = 10
# sch_reduceLR_threshold = 0.0001
# sch_reduceLR_threshold_mode = "rel"
# sch_reduceLR_cooldown = 0
# sch_reduceLR_min_lr = 0
# sch_reduceLR_eps = 1e-08

def create_dataloaders(mean, std, cuda, augment_func = "albumentation_augmentation"):
    
    ## Define data transformations
    train_transforms, test_transforms = eval("data_handling."+augment_func+"(mean, std)")

    ## Download & return transformed datasets
    trainset, testset = data_handling.return_datasets(train_transforms, test_transforms)

    ## Define data loaders
    trainloader, testloader = data_handling.return_dataloaders(trainset, testset, cuda, gpu_batch_size = 128, cpu_batch_size = 64)
    
    return trainloader, testloader


def trigger_training(model, device, trainloader, testloader, optimizer_name = "Adam", scheduler_name = "OneCycle", criterion_name = "CrossEntropyLoss", lambda_l1 = 0, epochs = 100):
    
    train_acc, train_losses, test_acc, test_losses, lrs = [], [], [], [], []
    
    if (optimizer_name == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr = standard_lr, weight_decay = L2_penalty)
    else:
        optimizer = optim.SGD(model.parameters(), lr = standard_lr, momentum = momentum_val)
        
    if (scheduler_name == "OneCycle"):
        scheduler = OneCycleLR(optimizer, max_lr = standard_lr, epochs = epochs, steps_per_epoch = len(trainloader))
    elif (scheduler_name == "ReduceLROnPlateau"):
        scheduler = ReduceLROnPlateau(optimizer, mode = sch_reduceLR_mode, factor = sch_reduceLR_factor, patience = sch_reduceLR_patience, threshold = sch_reduceLR_threshold, threshold_mode = sch_reduceLR_threshold_mode, cooldown = sch_reduceLR_cooldown, min_lr = sch_reduceLR_min_lr, eps = sch_reduceLR_eps, verbose = False)
    elif (scheduler_name == "None"):
        scheduler = "None"
        
    if (criterion_name == "CrossEntropyLoss"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = F.nll_loss
     
    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        train.train(model, device, trainloader, train_acc, train_losses, optimizer, scheduler, criterion, lrs, lambda_l1)
        test.test(model, device, testloader, test_acc, test_losses, criterion)
    
    
    return train_acc, train_losses, test_acc, test_losses, lrs


