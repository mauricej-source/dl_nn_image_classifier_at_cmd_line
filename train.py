#-- ---------------------------------------------------------------------------------------
# Udacity Image Classifier Project - Part 2: Requirements
# *Useful Links: https://docs.python.org/3/library/argparse.html
#                https://pymotw.com/3/argparse/
# Developer:  Maurice Johnson
# Date:       05/04/2020
#-- ---------------------------------------------------------------------------------------
#
#train.py
#  - scope:   will train a new network on a dataset and save the model as a checkpoint
#  - output:  training loss, validation loss, and validation accuracy as the network trains
#  - options: Argument 1:  data directory name
#             - usage:   python train.py {argument 1 => data directory name}
#            
#            Optional - Set Default Values if not provided
#             Argument 2:  directory to save checkpoint to
#            - usage:   python train.py data_directory --save_dir {argument 2 => destination directory to save model checkpoint} 
#            
#             Argument 3:  choice of architecture
#            - usage:   python train.py data_directory --arch {argument 3 => architecture} 
#            
#             Argument 4:  Set hyperparameter - learning_rate
#            - usage:   python train.py data_directory --learning_rate {argument 4 => 0.01}              
#            
#             Argument 5:  Set hyperparameter - hidden_units
#            - usage:   python train.py data_directory --hidden_units {argument 5 => 512} 
#
#             Argument 6:  Set hyperparameter - epochs
#            - usage:   python train.py data_directory --epochs {argument 6 => 20}           
#
#             Argument 7:  Set device - gpu
#            - usage:   python train.py data_directory --device {argument 7 => gpu/cpu} 
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
from collections import OrderedDict

#-- ---------------------------------------------------------------------------------------
#   Declarations
#-- ---------------------------------------------------------------------------------------
input_file_directory = None
model_checkpoint_directory = None
model_architecture = None
model_learning_Rate = 0
model_hidden_units = 0
__device_STRING = None
__device_available__ = None
num_of_desired_outputs = 0
num_of_input_features = 0
model_batch_size = 0
model_epochs = 0
cust_hidden_units = False

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
        
def get_model(architecture_name, freeze_parameters):
    print("Initiating the Neural Network Model...\n")
    
    trained_model = None
    cleared_for_training = False
    
    #Load a Pretrained Network - Default vgg16
    if architecture_name == 'vgg16':
        trained_model = models.vgg16(pretrained=True)
        trained_model.name = 'vgg16'
        cleared_for_training = True
    elif architecture_name == 'resnet18':
        trained_model = models.resnet18(pretrained=True)
        trained_model.name = 'resnet18'
        cleared_for_training = True
    elif architecture_name == 'alexnet':
        trained_model = models.alexnet(pretrained=True)
        trained_model.name = 'alexnet'
        cleared_for_training = True
    elif architecture_name == 'squeezenet1':
        trained_model = models.squeezenet1_0(pretrained=True)
        trained_model.name = 'squeezenet1'
        cleared_for_training = False
    elif architecture_name == 'densenet161':
        trained_model = models.densenet161(pretrained=True)
        trained_model.name = 'densenet161'
        cleared_for_training = False
    elif architecture_name == 'inception':
        trained_model = models.inception_v3(pretrained=True)
        trained_model.name = 'inception_v3'
        cleared_for_training = False
    elif architecture_name == 'googlenet':
        trained_model = models.googlenet(pretrained=True)
        trained_model.name = 'googlenet'
        cleared_for_training = False
    elif architecture_name == 'shufflenet':
        trained_model = models.shufflenet_v2_x1_0(pretrained=True)
        trained_model.name = 'shufflenet_v2_x1_0'
        cleared_for_training = False
    elif architecture_name == 'mobilenet':
        trained_model = models.mobilenet_v2(pretrained=True)
        trained_model.name = 'mobilenet_v2'
        cleared_for_training = False
    elif architecture_name == 'resnext50':
        trained_model = models.resnext50_32x4d(pretrained=True)
        trained_model.name = 'resnext50_32x4d'
        cleared_for_training = False
    elif architecture_name == 'wide_resnet50':
        trained_model = models.wide_resnet50_2(pretrained=True)
        trained_model.name = 'wide_resnet50_2'  
        cleared_for_training = False
    elif architecture_name == 'mnasnet1':
        trained_model = models.mnasnet1_0(pretrained=True)
        trained_model.name = 'mnasnet1_0'    
        cleared_for_training = False
    else:
        print("train.py - Function: get_model\nERROR:  Unknown model passed at the command line...")
    
    if (cleared_for_training == True) and (freeze_parameters == True):
        #Freeze parameters to prevent training
        for param in trained_model.parameters():
             param.requires_grad = False

    return trained_model, cleared_for_training

def get_model_input_features(architecture_name, trained_model, is_custom_hidden_units):
    num_of_input_features = 0
    
    #Default vgg16
    if architecture_name == 'vgg16':      
        if is_custom_hidden_units:
            #Number of Input Features
            num_of_input_features = trained_model.classifier[0].in_features
        else:
            #Number of Input Features
            num_of_input_features = trained_model.classifier[6].in_features
    elif architecture_name == 'alexnet':
        if is_custom_hidden_units:
            #Number of Input Features
            num_of_input_features = trained_model.classifier[1].in_features
        else:
            #Number of Input Features
            num_of_input_features = trained_model.classifier[6].in_features
    elif architecture_name == 'resnet18':
        #Number of Input Features
        num_of_input_features = trained_model.fc.in_features
    else:
        print("train.py - Function: get_model_input_features.")
        print("ERROR:  Unknown model passed at the command line...")
        
    return num_of_input_features
            
def get_classifier(trained_model, architecture_name, model_hidden_units, num_of_desired_outputs):   
    # A new feedforward network is defined for use as a classifier using the features as input
    # Validate the Number of Output Features for this Trained Model
    #
    # NOTE:  From before we are seeking 102 as noted in the Classes Count
    #        and the Category Count
    #print(trained_model.classifier[6].out_features) #1000
    
    print("Initiating the Neural Network Model Classifier...\n")
    
    num_of_input_features = 0
    features = None
    trained_classifier = None
    _hidden_layers = False
    
    if (model_hidden_units >= 0) and (model_hidden_units <= 1):
        #print("Creating Default Classifier...")
  
        #Number of Input Features
        num_of_input_features = get_model_input_features(architecture_name, trained_model, False)
    
        #Default vgg16
        if (architecture_name == 'vgg16') or (architecture_name == 'alexnet'):       

            #Chomp the layers of the Trained Model
            features = list(trained_model.classifier.children())[:-1]

            # Dynamically extend the output layer with number of outputs
            features.extend([nn.Linear(num_of_input_features, num_of_desired_outputs)]) 

            # Replace the trained model classifier with our designated output layer size
            trained_classifier = nn.Sequential(*features)
            
            trained_model.classifier = trained_classifier  
            
        elif architecture_name == 'resnet18':            
            trained_classifier = nn.Linear(num_of_input_features, num_of_desired_outputs)
            trained_model.fc = trained_classifier
        else:
            print("train.py - Function: get_classifier")
            print("ERROR:  Unknown model passed at the command line...") 
    else:
        if model_hidden_units > num_of_desired_outputs:
            #Declare required Hidden Layer Attributes
            _hidden_layers = True
            
            #Number of Input Features
            num_of_input_features = get_model_input_features(architecture_name, trained_model, True)   

            if model_hidden_units < num_of_input_features:
                #Default vgg16
                if architecture_name == 'vgg16': 
                    trained_classifier = nn.Sequential(nn.Linear(num_of_input_features,model_hidden_units),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(0.5),
                                                       nn.Linear(model_hidden_units,model_hidden_units),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(0.5),
                                                       nn.Linear(model_hidden_units, num_of_desired_outputs))
                                                       
                    trained_model.classifier = trained_classifier   
                elif architecture_name == 'alexnet':  
                    trained_classifier = nn.Sequential(nn.Dropout(0.5), 
                                                       nn.Linear(num_of_input_features,model_hidden_units),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(0.5),
                                                       nn.Linear(model_hidden_units,model_hidden_units),
                                                       nn.ReLU(inplace=True),
                                                       nn.Linear(model_hidden_units, num_of_desired_outputs))
                    
                    trained_model.classifier = trained_classifier                   
                elif architecture_name == 'resnet18': 
                    print("train.py - Function: get_classifier")
                    print("WARNING:  This model does not have any hidden units to replace with the ")
                    print("          value passed at the command line: ", model_hidden_units)
                else:
                    print("train.py - Function: get_classifier")
                    print("ERROR:  Unable to construct classifier with the Model ")
                    print("        Unknown model passed at the command line: ", model_hidden_units)
            else:
                error_stmt_one = "\nERROR:  The Model Hidden Units {mhu:}, passed at the command line, must be "
                error_stmt_two = "less than the size of the input features {noif:}"

                print("train.py - Function: get_classifier")
                print(error_stmt_one.format(mhu = model_hidden_units))
                print(error_stmt_two.format(noif = num_of_input_features))
                
            #Validate Hidden Units substituted within Classifier                                
            #print(trained_classifier)               
        else:
            error_stmt_one = "\nERROR:  The Model Hidden Units {mhu:}, passed at the command line, must be "
            error_stmt_two = "greater than the desired output layer size {ols:}"

            print("train.py - Function: get_classifier")
            print(error_stmt_one.format(mhu = model_hidden_units))
            print(error_stmt_two.format(ols = num_of_desired_outputs))
            
    return trained_model, trained_classifier, num_of_input_features, _hidden_layers

def get_optimizer(model, architecture_name, learning_rate):
    optimizer = None
    
    #Default vgg16
    if (architecture_name == 'vgg16') or (architecture_name == 'alexnet'):       
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
                                  model.classifier.parameters()), lr=learning_rate, momentum=0.9)
    elif architecture_name == 'resnet18':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
                                  model.parameters()), lr=learning_rate, momentum=0.9)
    else:
        print("train.py - Function: get_optimizer.")
        print("ERROR:  Unknown model passed at the command line...")
        
    return optimizer

def train_the_model(model, architecture_name, device, learning_rate, epochs, dataimageloader, testloader):
    print("Initiating the Neural Network Model Training.  Please stand by...\n")
    #Establish the Loss Function
    loss_criterion = nn.CrossEntropyLoss()
     
    #Establish the Optimizer - Filter out the Frozen Parameters Only train the classifier parameters
    optimizer = get_optimizer(model, architecture_name, learning_rate)
     
    #Validate Model/Optimizer
    #print(model)
    #print(optimizer)
     
    #Move the model to the designate device
    model.to(device)
     
    #Proceed to Train the New Neural Network
    steps = 0
    running_loss = 0
    cycle_every = 25
     
    #Transition the Model to the Training Mode
    model.train()
         
    for epoch in range(epochs):  
         
        for images, labels in dataimageloader:
            steps += 1
               
            #Move images and labels to designate device
            images,labels = images.to(device), labels.to(device)    
             
            #Validate type, size, dim
            #print(type(images))  #<class 'torch.Tensor'>
            #print(type(labels))  #<class 'torch.Tensor'>
            #print(images.size()) #torch.Size([64, 3, 224, 224])
            #print(images.dim())  #4
             
            # zero the parameter gradients 
            #(Time elapsed at this step: 71016.375 millisecs)
            optimizer.zero_grad()
             
            # devise the output 
            #(Time elapsed at this step: 80399.429 millisecs)
            logps = model.forward(images)
               
            # calculate the loss 
            #(Time elapsed at this step: 79864.281 millisecs)  
            loss = loss_criterion(logps, labels)
               
            # backwards prop - aggregate the gradients based on params 
            #(Time elapsed at this step: 79782.492 millisecs)  
            loss.backward()
               
            # update the params based upon gradient 
            #(Time elapsed at this step: 79896.265 millisecs)  
            optimizer.step()
             
            running_loss += loss.item()
            
            #Test Network accuracy
            if steps % cycle_every ==0:
                 #Transition the Model to Evaluation Mode
                      model.eval()
                       
                      test_loss = 0
                       
                      accuracy = 0
                       
                      #Validation Loop
                      for images,labels in testloader:
                           images,labels = images.to(device), labels.to(device)
                           
                           logps = model.forward(images)
                            
                           loss = loss_criterion(logps,labels)
                        
                           test_loss += loss.item()
                           
                           #calculate our accuracy
                           ps = torch.exp(logps)
                           
                           #Top Probabilities and Top Classes along columns
                           top_ps,top_class = ps.topk(1,dim=1)
                           
                           equality = top_class == labels.view(*top_class.shape)
                           
                           accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                            
                      #Report Testing Results
                      print(f"Epoch {epoch+1}/{epochs}..)"
                            f"Train Loss: {running_loss/cycle_every:.3f}.. "
                            f"Test Loss: {test_loss/len(testloader):.3f}.. "
                            f"Test Accuracy: {accuracy/len(testloader):.3f}")
                             
                      running_loss = 0
                       
                      #Transition the Model back to Training Mode
                      model.train()
    
    print('\nFinished Training, Testing, and Validating...\n')
     
    if torch.cuda.is_available():
        # Releases all unoccupied cached memory currently held by
        # the caching allocator so that those can be used in other
        # GPU application and visible in nvidia-smi
        torch.cuda.empty_cache()
        
    return model, optimizer, loss, epochs
 
def store_model_checkpoint(trained_model, model_checkpoint_directory, num_of_input_features, \
                           num_of_desired_outputs, trained_classifier, image_datasets, \
                           model_batch_size, trained_epochs, trained_optimizer, trained_loss, \
                           is_custom_units):
    print("Initiate storing the Neural Network Model Checkpoint...\n")
    
    if os.path.isdir(model_checkpoint_directory):
        print('SUCCESS:  Model Checkpoint Destination Directory Found: ', model_checkpoint_directory)
    else:
        model_checkpoint_directory = os.getcwd()
        print('WARNING:  Either the Model Checkpoint Destination Directory was not found \
        or not provided.  Will use Current Working Directory in its stead: ', model_checkpoint_directory)
        
    #Get a new Model
    new_model, cleared_for_training = get_model(model_architecture, False)
 
    #Saving the Checkpoint will be unique to the model architecture...
    if(is_custom_units == False):
        checkpoint_file_name = trained_model.name + '_model_chckpnt.pt'
    else:
        checkpoint_file_name = trained_model.name + '_mdlcu_chckpnt.pt'
    
    #If file already exits remove it
    checkpoint_file_path = model_checkpoint_directory + os.path.sep + checkpoint_file_name
    if os.path.exists(checkpoint_file_path):
        os.remove(checkpoint_file_path)
    else:
        print("Can not delete the file as it doesn't exists")
        
    torch.save({
                'input_size': num_of_input_features,
                'output_size': num_of_desired_outputs,
                'epoch': trained_epochs,
                'batch_size': model_batch_size,
                'model': new_model,
                'classifier': trained_classifier,
                'optimizer': trained_optimizer.state_dict(),
                'state_dict': trained_model.state_dict(),
                'loss': trained_loss,
                'class_to_idx': image_datasets.class_to_idx
                }, checkpoint_file_path)

    print("------------------------------------------------------------------")    
    print('Trained Model Checkpoint Successfully Stored:')
    print("------------------------------------------------------------------")
    print('File Path    :', checkpoint_file_path)
    print('Access time  :', time.ctime(os.path.getatime(checkpoint_file_path)))
    print('Modified time:', time.ctime(os.path.getmtime(checkpoint_file_path)))
    print('Change time  :', time.ctime(os.path.getctime(checkpoint_file_path)))
    print('Size         :', os.path.getsize(checkpoint_file_path))
    print("------------------------------------------------------------------")
    #print("  Input Size           : ", num_of_input_features)
    #print("  Output Size          : ", num_of_desired_outputs)
    #print("  Number of Epochs     : ", trained_epochs)
    #print("  Model Batch Size     : ", model_batch_size)
    #print("  Model Arch           : ", new_model.name)
    #print("  Classifier           : \n", trained_classifier)
    #print("  Optimizer            : \n", trained_optimizer.state_dict())
    #print("  State Dictionary     : \n", trained_model.state_dict())    
    #print("  Loss                 : ", trained_loss)    
    #print("  Image Classification : \n", image_datasets.class_to_idx)  
    #print("------------------------------------------------------------------")
    
#-- ---------------------------------------------------------------------------------------
#   Main - Parse Command Line Arguments
#-- ---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='A Program to train a New Neural Network based upon \
                                 the required data input directory passed at the command line...')
parser.add_argument("data_input_directory", help="Data Input Directory Required as first argument. \
     Data is currently located within the /home/workspace/ImageClassifier/flowers Directory beneath \
     one of three sub-folders (test / train / valid).")
parser.add_argument("--save_dir", default=os.getcwd(), help="Destination directory to save trained model checkpoint. \
     Default Value: Current directory as the train.py program")
parser.add_argument("--arch", type = str, default="vgg16", help="Model Architecture to be selected. \
     Default Value: 'vgg16'  Architecture Options: (vgg16, resnet18, alexnet)  \
     Yet to be added: (squeezenet1, densenet161, inception, googlenet, shufflenet, mobilenet, resnext50, \
     wide_resnet50, mnasnet1)")
parser.add_argument("--learning_rate", type = float, default=0.001, help="Model Training Learning Rate. \
     Default Value: 0.001")
parser.add_argument("--hidden_units", type = int, help="Number of Model Hidden Layers. \
     Default Value: None")
parser.add_argument("--epochs", type = int, default=10, help="Number of Epochs/Iterations to train model against. \
     Default Value: 10")
parser.add_argument("--device", type = str, default="gpu", help="Device chosen to perform training on. \
     Default Value: gpu")
parser.add_argument("--batch_size", type = int, default=64, help="Size of Batch during Model Training. \
     Default Value: 64")

args = parser.parse_args()

#-- ---------------------------------------------------------------------------------------
#   Main - Compensate for optional arguments passed at the command line
#-- ---------------------------------------------------------------------------------------
if args.data_input_directory:
    input_file_directory = args.data_input_directory
    #print("Input Data Directory passed at the command line: ", input_file_directory)
    
#-- ---------------------------------------------------------------------------------------
#   Main - Compensate for optional arguments passed at the command line
#-- ---------------------------------------------------------------------------------------
if args.save_dir:
    model_checkpoint_directory = args.save_dir
    #print("Destination directory to save trained model checkpoint: ", model_checkpoint_directory)
if args.arch:
    model_architecture = args.arch
    #print("Model Architecture to be selected: ", model_architecture)
if args.learning_rate:
    model_learning_Rate = args.learning_rate
    #print("Model Training Learning Rate: ", model_learning_Rate)
if args.hidden_units != None:
    model_hidden_units = args.hidden_units
    #print("Number of Model Hidden Layers: ", model_hidden_units)
    #print(type(model_hidden_units))
if args.epochs:
    model_epochs = args.epochs
    #print("Number of Epochs/Iterations to train model against: ", model_epochs)
if args.device:
    __device_STRING = args.device
    #print("Device chosen to perform training on: ", __device_STRING) 
    __device_available__ = get_device(__device_STRING)
    #print("Device Determined to be Available: ", __device_available__)
if args.batch_size:
    model_batch_size = args.batch_size
    #print("Size of Batch during Model Training: ", model_batch_size)    
#-- ---------------------------------------------------------------------------------------
#   Main - Validate Data Input Directory
#-- ---------------------------------------------------------------------------------------    
if os.path.isdir(input_file_directory):
    #print('SUCCESS:  Data Input File Directory Found...')
    if os.path.isdir(model_checkpoint_directory):
        #print('SUCCESS:  Modal Checkpoint Destination Directory Found...')
    
        #-- -----------------------------------------------------------------------------------
        # - Apply Transformations:
        #    - Random Scaling
        #    - Cropping
        #    - Flipping
        #
        #   This will help the network generalize leading to better performance.
        #
        # - Also  resize images to 224x224 pixels as required by pre-trained networks
        #
        # - The pre-trained networks you'll use were trained on the ImageNet dataset 
        #   where each color channel was normalized separately. For all three sets 
        #   you'll need to normalize the means and standard deviations of the images 
        #   to what the network expects
        #   - For the means, it's [0.485, 0.456, 0.406]
        #   - For the standard deviations [0.229, 0.224, 0.225]
        #   - These values will shift each color channel to be centered at 0 and range 
        #     from -1 to 1.
        #  
        #-- ----------------------------------------------------------------------------------- 
        # Training data augmentation - Data normalization
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
        
        test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        
        validate_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        
        train_dir = input_file_directory + '/train'
        valid_dir = input_file_directory + '/valid'
        test_dir = input_file_directory + '/test'
        
        # Data loading
        image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
        test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
        validate_datasets = datasets.ImageFolder(valid_dir, transform=validate_transforms)
        
        # Data batching
        dataimageloader = torch.utils.data.DataLoader(image_datasets, batch_size=model_batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_datasets, batch_size=model_batch_size)
        validateloader = torch.utils.data.DataLoader(validate_datasets,batch_size=model_batch_size)
        
        #Validate loading of datasets
        #print(image_datasets[10])
        #print(test_datasets[10])
        #print(validate_datasets[10])
        
        #Validate dataloaders
        #print("Image Data Loader Shape: ", dataimageloader.dataset[0][0].shape) #3,224,224
        #print("Image Data Loader Size: ", dataimageloader.dataset[0][0].size()) #3,224,224
        #print("Image Dataset Type [0][0]: ", type(dataimageloader.dataset[0][0])) #Torch Tensor
        #print("Image Dataset Type [0][0]: ", type(dataimageloader.dataset[0][1])) #int
        
        #How many Classes are there? - 102
        #print("Overall Number of Flower Classifications: ", len(image_datasets.classes))  
        
        #Validate loading of datasets
        #print("Image Dataset Sampling: \n", image_datasets[10])
        
        #-- -----------------------------------------------------------------------------------
        # - Load a pre-trained network (If you need a starting point, the VGG networks work great 
        #   and are straightforward to use)
        #
        # - Define a new, untrained feed-forward network as a classifier, using ReLU activations 
        #   and dropout
        #
        # - Train the classifier layers using back-propagation using the pre-trained network to get 
        #   the features
        #
        # - Track the loss and accuracy on the validation set to determine the best hyperparameters
        #-- -----------------------------------------------------------------------------------         
        #Get the Model
        trained_model, cleared_for_training = get_model(model_architecture, True)
        
        #Validate if model has been cleared for training
        if cleared_for_training == True:
            #Validate the Model Architecture Name
            #print("Model Architecture Name Selected: ", trained_model.name)
            
            #Retrieve the number of outputs required
            num_of_desired_outputs = len(image_datasets.classes)
                            
            #Conditionally test for type and value:  model_hidden_units, num_of_desired_outputs
            if (type(num_of_desired_outputs) is int) and (num_of_desired_outputs > 0):
                if (type(model_hidden_units) is int) and (model_hidden_units >= 0):
                    #Get the Classifier, Model with Classifier Attached, and Number of Input Features
                    trained_model, trained_classifier, num_of_input_features, cust_hidden_units = get_classifier(trained_model, \
                                                        model_architecture, model_hidden_units, num_of_desired_outputs)
                    
                    #Validate Both Classifier and Model
                    #print(trained_classifier)
                    #print(trained_model)
                    
                    #Conditionally test for type and value:  num_of_input_features
                    if (type(num_of_input_features) is int) and (num_of_input_features > 0):
                        #Conditionally test for type and value:  model_learning_Rate and model_epochs
                        if (type(model_learning_Rate) is float) and (model_learning_Rate > 0):
                            if (type(model_epochs) is int) and (model_epochs > 0):
                                if trained_classifier != None:
                                    #-- -----------------------------------------------------------------------------------
                                    # - The parameters of the feedforward classifier are appropriately trained, 
                                    #       while the parameters of the feature network are left static
                                    # - The network's accuracy is measured on the test data
                                    # - During training, the validation loss and accuracy are displayed
                                    #-- ----------------------------------------------------------------------------------- 
                                    trained_model, trained_optimizer, trained_loss, trained_epochs = train_the_model(trained_model, \
                                                                    model_architecture, __device_available__, model_learning_Rate, \
                                                                    model_epochs, dataimageloader, testloader)
                                    
                                    #-- -----------------------------------------------------------------------------------
                                    # - There is a function that successfully loads a checkpoint and rebuilds the model
                                    # - The trained model is saved as a checkpoint along with associated hyperparameters 
                                    #       and the class_to_idx dictionary
                                    #-- -----------------------------------------------------------------------------------                             
                                    store_model_checkpoint(trained_model, model_checkpoint_directory, num_of_input_features, \
                                                           num_of_desired_outputs, trained_classifier, image_datasets, \
                                                           model_batch_size, trained_epochs, trained_optimizer, trained_loss, \
                                                           cust_hidden_units)
                            else:
                                print('ERROR:  The model training epochs, passed at the command line, is either')
                                print('        not numeric or is less than or equal to zero: ', model_epochs)                    
                        else:
                            print('ERROR:  The model learning rate, passed at the command line, is either')
                            print('        not a floating decimal or is less than or equal to zero:', model_learning_Rate)                  
                    else:
                        print('ERROR:  The model''s determined input layer units value is either')
                        print('        not numeric or is less than or equal to zero: ', num_of_input_features) 
                else:
                    print('ERROR:  The model hidden units value, passed at the command line, is either')
                    print('        not numeric or is less than zero: ', model_hidden_units)                    
            else:
                print('ERROR:  The model''s determined output layer units value is either, not numeric')
                print('        or is less than or equal to zero: ', num_of_desired_outputs)
        else:
            print('ERROR:  The model selected has either resulted in error OR has yet to be cleared for ')
            print('        training within this program: ', model_architecture)
    else:
        print('ERROR:  Modal Checkpoint Destination Directory NOT Found: ', model_checkpoint_directory)         
else:
    print('ERROR:  Data Input File Directory NOT Found: ', input_file_directory)
