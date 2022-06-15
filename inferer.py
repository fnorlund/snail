import os
import numpy as np
import torch
#import torch.nn.functional as F
import torchvision
#from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from zmq import device

#from utils import _create_model_training_folder
#from data.transforms import get_single_transform, transfer_transforms

class BYOLTransferInferer:
    def __init__(self, online_network, dataset, q_transforms, device, **kwargs):
        self.kwargs = kwargs
        self.online_network = online_network
        #self.best_model_wts = online_network
        self.val_dataset = dataset
        self.device = device
        self.trans = q_transforms(
            self.kwargs['transforms']['input_shape'], 
            self.kwargs['transforms']['s']) 
        self.writer = SummaryWriter()
        #self.m = params['m']
        self.batch_size = 1 # params['batch_size']
        self.num_workers = 1 # params['num_workers']

    def infer(self):
        
        self.online_network.eval()
        
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)
        
        i = 0
        with torch.no_grad():
            for (batch_view_1, label) in val_loader:
                batch_view_1, label = batch_view_1.to(self.device), label.to(self.device)
                y = self.online_network(batch_view_1)
                filename = self.val_dataset.datasets[0].imgs[i]
                print("Filename: {}".format(filename))
                print("Class: {}, Label={}".format(y, label))
                print()
                i += 1
            return y
    
    def infer_from_path(self, img_path):
        
        self.online_network.eval()
        torch.inference_mode(mode=True)
        q = torch.is_inference_mode_enabled()
        #with torch.no_grad():
        for filename in os.listdir(img_path):
            img_unchg = torchvision.io.read_image(os.path.join(img_path,filename), mode=torchvision.io.ImageReadMode.UNCHANGED)
            #img_unchg = torchvision.io.decode_image(img_unchg)
            img_unchg = torchvision.io.read_image(os.path.join(img_path,filename), mode=torchvision.io.ImageReadMode.RGB) # image type = tensor
            to_pil = torchvision.transforms.ToPILImage() 
            pil_img = to_pil(img_unchg)
            #img_np = np.array(pil_img)
  
            trfs = self.trans.transforms # Pointer to instantiated transform objects
            img_transformed = trfs[0](pil_img) # Resize with PIL-image -> (640,640)
            img_transformed = np.array(img_transformed) # to np-array -> (640,640)
            img_transformed = trfs[1](img_transformed) # ToTensor+scaling [0,1] -> (3,640,640)
            img_transformed = trfs[2](img_transformed)
            img_transformed = img_transformed.expand(1,-1,-1,-1)

            if self.device == 'cuda':
                img_transformed = img_transformed.cuda() # Convert to torch.cuda.FloatTensor

            y = self.online_network(img_transformed)
            y_distance = torch.abs(y[0][0] - y[0][1])
            value, class_ind = torch.max(y,1)
            pred = 'slug' if class_ind[0] == 0 else 'not_slug'
            print("File: {}. Val: {}. Pred: {}. Dist: {}".format(filename, y.data, pred, y_distance))

        '''
        with open(os.path.join(img_path, filename), 'r') as f:
            img_unchg = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.UNCHANGED)
            img_tensor = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # image type = tensor
            #text = f.read()
            #print(text)
        '''