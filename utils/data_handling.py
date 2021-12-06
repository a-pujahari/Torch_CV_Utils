from torchvision import datasets, transforms
import torchvision
import torch
import numpy as np


cifar10_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Calculate Dataset Statistics
def return_dataset_statistics():
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train = True, download = True, transform = train_transform)
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255

    return mean, std


def return_datasets(train_transforms, test_transforms):
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transforms)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transforms)
    
    return trainset, testset

def return_dataloaders(trainset, testset, cuda, gpu_batch_size = 128, cpu_batch_size = 64):
    
    dataloader_args = dict(shuffle = True, batch_size = gpu_batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle = True, batch_size = cpu_batch_size)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return trainloader, testloader

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
## Function to show sample data
def show_sample_data(trainloader, num_images = 16):
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images[0:num_images]))
    # print labels
    print(' '.join('%5s' % cifar10_classes[labels[j]] for j in range(num_images)))
