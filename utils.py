import os
import torch
from torch.nn import Linear, Sequential
#from torchvision.models import resnet18
from shutil import copyfile


def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def create_transfer_model(model_name):
    out_features = 2
    
    if model_name == 'resnet18':
        # check if model already downloaded
        try:
            preTrainedModel = load_model(relative_path='pretrained_network/' ,filename=model_name + '.pth')
        except:
            # Download model
            preTrainedModel = resnet18(pretrained=True)

        # Freeze weights update
        for param in preTrainedModel.parameters():
            param.requires_grad = False

        # Replace last layer. Parameters of newly constructed layer (modules) have requires_grad=True by default
        seq = Sequential()
        num_ftrs = preTrainedModel.fc.in_features
        preTrainedModel.fc = Linear(num_ftrs, out_features)
        #preTrainedModel['fc.bias'].requires_grad = True
        #preTrainedModel['fc.weight'].requires_grad = True
        
        # Save models
        save_model(preTrainedModel, relative_path='pretrained_network/' ,filename='resnet18.pth')
        # 
        save_model_state_dict(preTrainedModel, relative_path='pretrained_network/' ,filename='resnet18.pt')

    return preTrainedModel

def view_image(image):
    pass


##### Print model layers, model state_dicts (weights), optimizer #####

def print_model_layers(model):
    # tex model.resnet.layer1[0].conv1
    pass

def print_model_dict(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    pass

def print_optimizer_dict(optimizer):
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])


####### ABOUT SAVE/LOAD ##########
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_model(model, relative_path, filename):
    # FN. pre-trained model
    #
    dir_name = os.path.dirname(__file__)
    torch.save(model,os.path.join(dir_name, relative_path, filename))
    #torch.save(model.state_dict(), os.path.join(dir_name, relative_path, filename))

def load_model(relative_path,filename):
    dir_name = os.path.dirname(__file__)
    model = torch.load(os.path.join(dir_name, relative_path, filename))
    return model

def save_model_state_dict(model, relative_path, filename):
    # FN. pre-trained model
    #
    dir_name = os.path.dirname(__file__)
    #torch.save(model,os.path.join(dir_name, relative_path, filename))
    #torch.save(model.state_dict(), os.path.join(dir_name, relative_path, filename))
    torch.save({'online_network_state_dict': model.state_dict()}, os.path.join(dir_name, relative_path, filename))

def load_model_state_dict(relative_path,filename):
    dir_name = os.path.dirname(__file__)
    model = torch.load(os.path.join(dir_name, relative_path, filename))
    return model