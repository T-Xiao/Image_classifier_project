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
import json
import argparse
parser = argparse.ArgumentParser(description='Predict an image')
# arguments to be added here: 
parser.add_argument('checkpoint', action='store', default='checkpoint.pth',help='Select a pretrained model')
parser.add_argument('image_path', action='store', default='./flowers/test/10/image_07090.jpg',help='test an image')
parser.add_argument('--top_k', action='store', default=5,help='Top k classes')
parser.add_argument('--category_names', action='store', default='cat_to_name.json',help='Category names')
parser.add_argument('--gpu',action='store_true', default= False, help='Use GPU')

args = parser.parse_args()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(image_path):
    checkpoint = torch.load(image_path,map_location=lambda storage, loc: storage)
    model_string = 'models.' + checkpoint['architecture'] + '(pretrained=True)'
    model = eval(model_string)    
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs=checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # load an image using PIL
    im = Image.open(image)
    ratio = 256/min(im.size)
    new_size = tuple([int(x*ratio) for x in im.size])
    im =im.resize(new_size, Image.ANTIALIAS)
    # center crop
    centercrop = transforms.CenterCrop((224,224)) 
    centercrop_image = centercrop(im)
    # convert color chanels to 0-1
    np_image = np.array(centercrop_image)/255
    # normalize the image by 
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-means)/std
    #reorder dimensions
    np_image = np_image.transpose(2, 0, 1)
    return np_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model):   
    img = process_image(image_path)
    model.to("cpu")
    img1 = torch.tensor(img).unsqueeze_(0).float()
    with torch.no_grad():
        output = model.forward(img1)
    ps = torch.exp(output)
    top_prob, top_class = ps.topk(int(args.top_k))
    aaa=top_class.tolist()[0]
    mapping = {classes: ids for ids, classes in
                model.class_to_idx.items()
                }
    top_class_index = [mapping [lab] for lab in aaa]
    top_class_name = [cat_to_name[lab] for lab in top_class_index]
    return top_prob,top_class_index,top_class_name


if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
# Flower name and class probability
model = load_checkpoint(args.checkpoint)
# label mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
probs, classes, flower_names=predict(args.image_path,model)
print (probs,flower_names)
 
    


