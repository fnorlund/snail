# export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
#from fastapi import FastAPI
import os
import sys
import copy
#from click import pass_context
import numpy as np
#from numpy import True_
sys.path.append('./')
import torch
torch.cuda.empty_cache()
from torch.optim import lr_scheduler, Adam
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
#from torchvision.models import resnet18
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18Transfer, ResNet18
from trainer import BYOLTransferTrainer, BYOLTrainer
import utils
from data.transforms import get_simclr_data_transforms, infer_transforms
from datamanager import SlugDataset

'''
learn_or_infer = '_' # 'train', 'infer', 'deploy'
training_mode = 'transfer_learning' # 'transfer_learning', 'contrastive_learning'
model2use = '_' # 'model.pth'=transfer trained; 'online_network_transfer'=new pretrained resnet18.pth model
save_new_model = '' # False, True

device = 'cuda' if torch.cuda.is_available() and learn_or_infer != 'deploy' else 'cpu'
noGPUs = torch.cuda.device_count()
print('Torch version:',torch.__version__)
#torch.manual_seed(0)
torch.seed()
'''
cfg = False

def main(cmds):
    import io
    a=b'\x0f'.hex()
    b=b'\x10'.hex()
    c=b'\x0f'
    d=b'\x10'
    int.from_bytes(b'\x0f','big')

    learn_or_infer = '_' # 'train', 'infer', 'deploy'
    training_mode = 'transfer_learning' # 'transfer_learning', 'contrastive_learning'
    model2use = '_' # 'model.pth'=transfer trained; 'online_network_transfer'=new pretrained resnet18.pth model
    save_new_model = '' # False, True

    '''
    device = 'cuda' if torch.cuda.is_available() and learn_or_infer != 'deploy' else 'cpu'
    noGPUs = torch.cuda.device_count()
    print('Torch version:',torch.__version__)
    #torch.manual_seed(0)
    torch.seed()
    '''
    print('Command arguments:', str(cmds))
    argc = cmds
    cfg = False
    debug = True
    
    # Continue to train the model 'model.pth' @'runs/pretrained_network/checkpoints/model.pth'
    # python3 main.py train transfer_learning model
    #argc = ['main.py', 'train', 'transfer_learning', 'model']

    # Create a new model, 'pretrained_network/resnet18.pth' and a copy 'runs/pretrained_network/checkpoints/model.pth'
    # python3 main.py train transfer_learning online_network_transfer
    #argc = ['main.py', 'train', 'transfer_learning', 'online_network_transfer']
    
    # Deploy to Torch Serve
    # python3 main.py deploy
    #argc = ['main.py', 'deploy']
    
    # Infer
    # python3 main.py infer
    #argc = ['main.py', 'infer']
    print("argc:",argc)

    if len(argc) > 1 and argc[1] == 'infer':
        import inferer
        learn_or_infer = 'infer'
        model2use = 'model.pth'
        cfg = True

    elif len(argc) > 1 and argc[1] == 'train':
        learn_or_infer = 'train'
        if argc[2] == 'transfer_learning':
            training_mode = 'transfer_learning'
            if argc[3] == 'model':
                model2use = 'model.pth'
                save_new_model = False
                cfg = True
            elif argc[3] == 'online_network_transfer':
                model2use = 'online_network_transfer'
                save_new_model = True
                cfg = True
            else:
                cfg = False
    elif len(argc) > 1 and argc[1] == 'deploy':
        learn_or_infer = 'deploy'
        model2use = 'model.pth'
        save_new_model = False
        cfg = True

    if not cfg:
        # Command examples
        # main.py train transfer_learning model
        # main.py train transfer_learning online_network_transfer
        # main.py infer
        # main.py deploy
        print('Debug or no such command. Ctrl-C to exit')
        print('Available arguments arg1, arg2, arg3')
        print('infer|train|deploy, transfer_learning, model|online_network_transfer')
        #exit(0)
    
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    #data_dir = './manual_download/local/fn'
    data_dir = './manual_download/local/fn_3_classes'
    #data_dir = './manual_download/local/q'

    device = 'cuda' if torch.cuda.is_available() and learn_or_infer != 'deploy' else 'cpu'
    noGPUs = torch.cuda.device_count()
    print('Torch version:',torch.__version__)
    #torch.manual_seed(0)
    torch.seed()

    
    if training_mode == 'contrastive_learning':
        data_transform = get_simclr_data_transforms(**config['data_transforms']) # Returns augmentation transforms

        train_dataset = datasets.STL10('./manual_download/', split='train+unlabeled', download=True,
                                    transform=MultiViewDataInjector([data_transform, data_transform]))

        # online network
        online_network = ResNet18(**config['network']).to(device)
        pretrained_folder = config['network']['fine_tune_from']
        #utils.save_model_state_dict(online_network,relative_path='pretrained_network/' ,filename='online_network.pt')

    if training_mode == 'transfer_learning':
        # FN: Create transfer model from pytorch (torchvision) pretrainded resnet18
        
        slugdataset = SlugDataset(data_dir=data_dir, **config['transfer_datasets'])
        online_network_transfer = ResNet18Transfer(**config['network']['transfer_network'])
        pretrained_folder = config['network']['fine_tune_from']
        #preTrainedModel = utils.create_transfer_model('resnet18')
        #preTrainedModel = resnet18(pretrained=True)
        if save_new_model:
            utils.save_model(online_network_transfer, relative_path='pretrained_network/' ,filename='resnet18.pt')
            utils.save_model_state_dict(online_network_transfer, relative_path='pretrained_network/' ,filename='resnet18.pth')
        #online_network_fn = utils.load_model_state_dict('pretrained_network/', 'resnet18.pt')
        pass
        #utils.print_model_dict(online_network_transfer)
        #utils.print_model_layers(online_network_transfer)
        #utils.print_model_layers(online_network_transfer) # NOT READY

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained model parameters
            if model2use == 'online_network_transfer': # start with new transfer training model
                if save_new_model:
                    path =  '/home/fredrik/source/snail/pretrained_network/resnet18.pth'
                    load_params = torch.load(path,
                                     map_location=torch.device(torch.device(device)))
                else:
                    load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'resnet18.pth')),
                                     map_location=torch.device(torch.device(device))) 
                online_network_transfer.load_state_dict(load_params['online_network_state_dict'])
                model = online_network_transfer
            
            elif model2use == 'online_network': # original network
                load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'online_network.pth')),
                                     map_location=torch.device(torch.device(device)))
                online_network.load_state_dict(load_params)
                model = online_network

            elif model2use == 'model.pth': # continue from previous transfer training runs
                # './runs/pretrained_network/checkpoints/model.pth'
                load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))
                online_network_transfer.load_state_dict(load_params['online_network_state_dict'])
                model = online_network_transfer


            #online_network.load_state_dict(load_params['online_network_state_dict'])
                

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    if device == 'cuda'and noGPUs > 0:
            device_ids = [0,1] #  [0,1]
            q=torch.cuda.current_device()
            nnModel = torch.nn.DataParallel(model, device_ids=device_ids) # Pick 1 or 2 GPU
            print("Let's use", len(device_ids), "GPUs!")
            for gpu in device_ids:
                print('Cuda device name:',torch.cuda.get_device_name([device_ids[gpu]]))
    model.to(device)
    
    if training_mode == 'contrastive_learning' and learn_or_infer == 'train':
        # predictor network
        predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                            **config['network']['projection_head']).to(device)

        # target encoder
        target_network = ResNet18(**config['network']).to(device)

        optimizer = torch.optim.SGD(list(target_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    if training_mode == 'contrastive_learning' and learn_or_infer == 'train':
        trainer = BYOLTrainer(online_network=online_network,
                            target_network=target_network,
                            optimizer=optimizer,
                            predictor=predictor,
                            device=device,
                            **config['trainer'])
        
    elif training_mode == 'transfer_learning' and learn_or_infer == 'train':
        #optimizer = torch.optim.SGD(model.resnet.fc.parameters(), **config['optimizer']['params'])
        optimizer = torch.optim.Adam(list(model.parameters()), **config['optimizer']['Adam']['params'])
        #optimizer = torch.optim.SGD(list(model.parameters()), **config['optimizer']['params'])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        if model2use == 'model.pth': # Continue with last saved optim params
            try: 
                optimizer.load_state_dict(load_params['optimizer_state_dict'])
            except:
                # Let trainer save optimizer_state_dict first run
                pass
            
        trainer = BYOLTransferTrainer(online_network=model,
                            target_network=None,
                            optimizer=optimizer,
                            scheduler=exp_lr_scheduler,
                            predictor=None,
                            device=device,
                            **config['trainer'])
        
        #trainer.train(slugdataset.image_train_datasets)
        trainer.train(slugdataset.image_datasets)

    elif learn_or_infer == 'infer':
        inferer = inferer.BYOLTransferInferer(model,slugdataset.image_val_datasets,infer_transforms,device,**config['transfer_datasets'])
        if inferer != None:
            path_to_infer_image = os.path.join(os.path.dirname(__file__), 'manual_download', 'infer_images')
            #inferer.infer(path_to_infer_image) #single image
            #path_to_infer_dirs = os.path.join(os.path.dirname(__file__), data_dir, 'train')
            path_to_infer_dirs = os.path.join(os.path.dirname(__file__), data_dir, 'val')
            inferer.infer_from_dirs(path_to_infer_dirs) #dirs
            #inferer.infer_from_path(path_to_infer_image) #single image
            #inferer.infer()
        else:
            pass
    
    elif learn_or_infer == 'deploy' and model2use == 'model.pth':
        load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))
        online_network_transfer.load_state_dict(load_params['online_network_state_dict'])
        model = online_network_transfer
        model.eval() # Set to evaluation/infer mode
        scripted_module = torch.jit.script(model)
        torch.jit.save(scripted_module, 'scripted_module.pth')
        torch.jit.save(scripted_module, os.path.join('./torchserve/mar_files','scripted_module.pth'))
        print('File scripted_module.pth is copied to folder torchserve\mar_files. Run torch-model-archiver')
        q = torch.jit.load('scripted_module.pth')

if __name__ == '__main__':
    
    main(sys.argv)
