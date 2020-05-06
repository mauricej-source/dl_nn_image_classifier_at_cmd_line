#-- ---------------------------------------------------------------------------------------
# Udacity Image Classifier Project - Part 2: Requirements
# *Useful Links: https://docs.python.org/3/library/argparse.html
#                https://pymotw.com/3/argparse/
# Developer:  Maurice Johnson
# Date:       05/04/2020
#-- ---------------------------------------------------------------------------------------
#              
#predict.py
#  - scope:   uses the model checkpoint to predict input images
#  - output:  Flower Name and Class Probability
#  - options: Argument 1:  Image File relative path + filename && Saved Model Checkpoint
#             - usage:   python predict.py /path/to/image checkpoint
#                        
#            Optional - Set Default Values if not provided
#            Argument 2:  Number of Probabilities to return
#             - usage:   python predict.py /path/to/image checkpoint --top_k {argument 3 => 3}
#                        
#            Argument 3:  Image Category JSON File
#             - usage:   python predict.py /path/to/image checkpoint --category_names {argument 4 => cat_to_name.json}  
#
#             Argument 4:  Set device - gpu
#            - usage:   python train.py data_directory --device {argument 5 => gpu/cpu}              
#-- ---------------------------------------------------------------------------------------
#--
#-- ---------------------------------------------------------------------------------------
#   Imports
#-- ---------------------------------------------------------------------------------------
import argparse
import os
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import json

#-- ---------------------------------------------------------------------------------------
#   Declarations
#-- ---------------------------------------------------------------------------------------
predict_image_filepath = None
model_checkpoint_filepath = None
predict_top_k = 0
predict_image_category_filepath = None
__device_STRING = None
__device_available__ = None

#Model Attributes
new_model = None 
last_epoch = 0 
input_size = 0 
output_size = 0 
optimizer = None
loss = 0.00 
class_to_idx = None

image_names = []
#-- ---------------------------------------------------------------------------------------
#   Define Functions
#-- ---------------------------------------------------------------------------------------
def get_device(device_chosen):
    device_determined = None
    
    print("Retrieving System Device...\n")
    
    if device_chosen != None:
        device_chosen = device_chosen.strip()
        #print("Device Chosen at command line: ", device_chosen)
        
        if len(device_chosen) > 0:
            if device_chosen.lower() == "gpu":
                if torch.cuda.is_available():
                    device_determined = torch.device("cuda")
                    
                    # Releases all unoccupied cached memory currently held by
                    # the caching allocator so that those can be used in other
                    # GPU application and visible in nvidia-smi
                    print("Releasing all unoccupied system device cached memory currently held...\n")
                    torch.cuda.empty_cache()
                else:
                    device_determined = torch.device("cpu")
            elif device_chosen.lower() == "cpu":
                device_determined = torch.device("cpu")
        else:
            print("train.py - Function: set_device - ERROR: Unknown device passed at the command line...")
    else:
        print("train.py - Function: set_device - ERROR: Unknown device passed at the command line...")
        
    return device_determined
    
def load_model_checkpoint(checkpoint_file_name):
    new_model = None 
    last_epoch = 0 
    input_size = 0 
    output_size = 0 
    optimizer = None
    loss = 0.00 
    class_to_idx = None
    
    find_the_model_list = checkpoint_file_name.split('_')
    architecture_name = find_the_model_list[0]
    
    model_checkpoint = torch.load(checkpoint_file_name)
        
    new_model = model_checkpoint['model']
    
    #Default vgg16
    if (architecture_name == 'vgg16') or (architecture_name == 'alexnet'):     
        #print("Architecture Name: ", architecture_name)
        
        new_model.classifier = model_checkpoint['classifier']
        
        #print("Attached Classifier: \n", new_model.classifier)
    elif architecture_name == 'resnet18':            
        new_model.fc = model_checkpoint['classifier']
    else:
        print("train.py - Function: load_model_checkpoint")
        print("ERROR:  Unknown model parsed from model checkpoint...") 
        
    #print("Loaded Model: \n", new_model)
        
    new_model.load_state_dict(model_checkpoint['state_dict'], strict=False)
    
    last_epoch = model_checkpoint['epoch']
        
    input_size = model_checkpoint['input_size']
        
    output_size = model_checkpoint['output_size']
        
    loss = model_checkpoint['loss']
        
    class_to_idx = model_checkpoint['class_to_idx']
                 
    optimizer = model_checkpoint['optimizer']
        
    for param in new_model.parameters():
        param.requires_grad = False
        
    return new_model, last_epoch, input_size, output_size, optimizer, loss, class_to_idx
    
def process_image(image):
    image_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = image_transformer(image)
    
    return image    

def predict(image_path, model, topk, device):
    image_in_question = Image.open(image_path)
    
    image_in_question = process_image(image_in_question)
        
    # Convert 2-Dimensional image to 1-Dimensional vector
    image_in_question = np.expand_dims(image_in_question, 0)
    
    image_in_question = torch.from_numpy(image_in_question)
    
    #Move model to device
    model.to(device)
    
    #Transition Model to Evaluation Mode
    model.eval()
    
    #Move Image to device
    inputs = Variable(image_in_question).to(device)
    
    #Pass Image into Model, caluculate raw output
    logits = model.forward(inputs)
    
    #Devise Probabilities from Raw Output
    probabilities = F.softmax(logits, dim=1)
    
    #Calculate the highest probabilities
    top_probabilities = probabilities.cpu().topk(topk)
    
    return (every_probality.data.numpy().squeeze().tolist() for every_probality in top_probabilities)

def get_image_names(json_file_path, flower_category, class_to_idx ):
    ret_image_names = []
    
    with open(json_file_path, 'r') as f:
        cat_to_name = json.load(f)
            
    #Validate loading of Category to Name Labels  
    #print(cat_to_name)
    
    for flower_key in flower_category:
        for image_key, image_value in class_to_idx.items():
            if image_value == flower_key:
                ret_image_names.append(cat_to_name[image_key])
                            
    return ret_image_names
#-- ---------------------------------------------------------------------------------------
#   Main - Parse Command Line Arguments
#-- ---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='A Program to predict what an image is, based upon \
                                 a Saved Neural Network Model Checkpoint and the required data \
                                 input directory, passed at the command line...')
parser.add_argument("image_filepath", help="Source File-path to retrieve the image that we would \
                    like to run a prediction against. \
                    Default Value: Current working directory as the predict.py program + os.path.sep + \
                    relative file-path to image: 'flowers/train/1/image_06734.jpg'")                               
parser.add_argument("model_checkpoint_filepath", help="Source File-path to retrieve the trained model checkpoint. \
                    Default Value: Current working directory as the predict.py program + os.path.sep + \
                    relative file-path to model checkpoint: 'vgg16_model_chckpnt.pt'")
parser.add_argument("--top_k", type = int, default=3, \
                    help="Top Number of Probabilities to return based upon the neural network \
                    model prediction. \
                    Default Value: 3")
parser.add_argument("--category_names", default=os.getcwd() + os.path.sep + 'cat_to_name.json', \
                    help="Source File-path to retrieve the Image Category JSON File. \
                    Default Value: Current working directory as the predict.py program + os.path.sep + \
                    relative file-path to JSON File: 'cat_to_name.json'")
parser.add_argument("--device", type = str, default="gpu", \
                    help="Device chosen to perform training on. \
Default Value: gpu")

args = parser.parse_args()  

#-- ---------------------------------------------------------------------------------------
#   Main - Compensate for optional arguments passed at the command line
#-- ---------------------------------------------------------------------------------------
if args.image_filepath:
    predict_image_filepath = args.image_filepath
    #print("Source File-path for the image that we wish to run prediction against: ", predict_image_filepath)
    
if args.model_checkpoint_filepath:
    model_checkpoint_filepath = args.model_checkpoint_filepath
    #print("Source File-path for the trained model checkpoint: ", model_checkpoint_filepath)
    
#-- ---------------------------------------------------------------------------------------
#   Main - Compensate for optional arguments passed at the command line
#-- ---------------------------------------------------------------------------------------
if args.top_k:
    predict_top_k = args.top_k
    #print("Top Number of Probabilities for Image Classification: ", predict_top_k)
if args.category_names:
    predict_image_category_filepath = args.category_names
    #print("Source File-path for the Image Category JSON File: ", predict_image_category_filepath)
if args.device:
    __device_STRING = args.device
    #print("Device chosen to perform training on: ", __device_STRING) 
    __device_available__ = get_device(__device_STRING)
    #print("Device Determined to be Available: ", __device_available__) 
    
#-- ---------------------------------------------------------------------------------------
#   Main - Validate the file-paths 
#-- ---------------------------------------------------------------------------------------    
if os.path.exists(predict_image_filepath):
    #print('SUCCESS:  Image File-path Found: ', predict_image_filepath)
    
    if os.path.exists(model_checkpoint_filepath):
        #print('SUCCESS:  Model Checkpoint File-path Found: ', model_checkpoint_filepath)
        
        if os.path.exists(predict_image_category_filepath):
            #print('SUCCESS:  Image Category JSON File-path Found: ', predict_image_category_filepath)  

            new_model, last_epoch, input_size, output_size, optimizer, loss, class_to_idx = \
             load_model_checkpoint(model_checkpoint_filepath)    

            #validate Checkpoint Load
            #print('Number of Epochs:', last_epoch)
            #print('Input Size      :', input_size)
            #print('Output Size     :', output_size)  
            #print('Loss            :', loss)          
            #print('Image Categories:\n', class_to_idx)             
            #print('Optimizer       :\n', optimizer) 
            #print('Model           :\n', new_model)
            
            if (type(predict_top_k) is int) and (predict_top_k > 0):
                #print('SUCCESS:  Number of Top Probabilities Value received: ', predict_top_k)  
                probabilities, flower_category = predict(predict_image_filepath, new_model, predict_top_k, __device_available__)
                
                image_classification_probability_stmt = "\nTop {top_probs:} Probabilities: "
                image_category_index_stmt = "Image File Category Index Lookup: "
                image_category_name_stmt = "Image File Category Name Lookup: "
                
                image_names = get_image_names(predict_image_category_filepath, flower_category, class_to_idx)
                
                print(image_classification_probability_stmt.format(top_probs = predict_top_k), probabilities)
                print(image_category_index_stmt, flower_category)
                print(image_category_name_stmt, image_names)
                print('\n')
            else:
                print('ERROR:  The prediction, top probabilities, passed at the command line, is either')
                print('        not numeric or is less than or equal to zero: ', predict_top_k) 
        else:
            print('ERROR:  Image Category JSON File-path NOT Found: ', predict_image_category_filepath)         
    else:
        print('ERROR:  Model Checkpoint File-path NOT Found: ', model_checkpoint_filepath) 
else:
    print('ERROR:  Image File-path NOT Found: ', predict_image_filepath)

       