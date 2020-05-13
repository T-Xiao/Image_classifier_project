#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports modules 
import numpy as np
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
from collections import OrderedDict

#Argument parser: 
parser = argparse.ArgumentParser(description='Train the image classifier')
# arguments to be added here: 
parser.add_argument('--arch', action='store', default='vgg16',help='Select a pretrained model')
parser.add_argument('--learning_rate', action='store', default=0.001,help='Select a learning rate')
parser.add_argument('--hidden_units1', action='store', default=4096, help='the first hidden unit, must be larger than hidden_unit2')
parser.add_argument('--hidden_units2', action='store', default=1024, help='the first hidden unit, must be smaller than hidden_unit1')
parser.add_argument('--epochs',action='store',default=2, help='Select an epoch')
parser.add_argument('--gpu',action='store_true', default= False, help='Use GPU')
args = parser.parse_args()



# Set the directory
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#  Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir,transform = train_transform) 
valid_dataset = datasets.ImageFolder(valid_dir,transform = valid_transform)
test_dataset = datasets.ImageFolder(test_dir,transform=valid_transform)
# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50,shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=50)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=50)

# label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build and train your network

# Use GPU if it's available
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

# Load the pretrained model from pytorch
model_string = 'models.' + args.arch + '(pretrained=True)'
model = eval(model_string)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, args.hidden_units1, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(args.hidden_units1, args.hidden_units2, bias=True)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(args.hidden_units2, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
# learning rates: 
optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 40
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss =0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")   
            running_loss = 0
            model.train()
            
# test dataset:
test_loss =0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print( f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")

# Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {'architecture':args.arch,
              'classifier': model.classifier,
              'epochs': args.epochs,
              'state_dict': model.state_dict(),
              'optimizer':model.state_dict(),
              'class_to_idx': model.class_to_idx,
              }

torch.save(checkpoint, 'checkpoint.pth')
