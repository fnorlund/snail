import yaml
from torchvision.models import resnet18
from handler import SlugDetect
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18    
    
# online network
config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
online_network = ResNet18(**config['network']).to(device)
projection = online_network.__dict__['_modules']['projetion']
projection = online_network.projetion


# FN: Download pretrainded model from pytorch (torchvision)
preTrainedModel = resnet18(pretrained=True)

# 2 ways of how to dig into model
first_layer_in_net = preTrainedModel.conv1 # use this way to replace layer as: model_conv.fc = nn.Linear(num_ftrs, 2)
first_layer_in_net = preTrainedModel.__dict__['_modules']['conv1']

layer4_0_conv1 = preTrainedModel.layer4[0].conv1
layer4_0_conv1 = preTrainedModel.__dict__['_modules']['layer4'][0].conv1

fc = preTrainedModel.__dict__['_modules']['fc']
fc = preTrainedModel.fc
scripted_module.resnet.fc.weight

# Client commands
curl -X POST "http://192.168.1.44:8080/predictions/multiple_data" -F 'file1=@/Users/harsh_bafna/test_images/kitten.jpg' -F 'file2=@/Users/harsh_bafna/test_images/kitten.jpg'

# Model create archive command, ie create the executable to run in torch server.
# Run command python3 main.py deploy
# https://github.com/pytorch/serve/blob/master/model-archiver/README.md
$ torch-model-archiver --model-name SlugDetect --version 0.1.0 --serialized-file torchserve/mar_files/scripted_module.pth --handler handler --runtime python3 --export-path torchserve/mar_files --force
# Don't use scripted file: --serialized-file torchserve/mar_files/scripted_module.pth
#$ torch-model-archiver --model-name slug-detect --version 0.1.0 --handler handler --runtime python3 --export-path torchserve/mar_files --force

# Register model, config etc in TorchServe command
$ torchserve --model-store torchserve/mar_files --workflow-store torchserve/mar_files/workflow-store --start --models SlugDetect=SlugDetect.mar --ts-config ./torchserve/config.properties
$ torchserve --stop

