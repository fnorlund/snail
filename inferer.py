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
    def __init__(self, online_network, dataset, infer_transforms, device, **kwargs):
        self.kwargs = kwargs
        self.online_network = online_network
        #self.best_model_wts = online_network
        self.val_dataset = dataset
        self.device = device
        self.trans = infer_transforms(
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
        
        s_correct = 0
        s_incorrect = 0
        i = 0
        for filename in os.listdir(img_path):
            #img_unchg = torchvision.io.read_image(os.path.join(img_path,filename), mode=torchvision.io.ImageReadMode.UNCHANGED)
            #img_unchg = torchvision.io.decode_image(img_unchg)
            img_unchg = torchvision.io.read_image(os.path.join(img_path,filename), mode=torchvision.io.ImageReadMode.RGB) # image type = tensor
            to_pil = torchvision.transforms.ToPILImage()
            pil_img = to_pil(img_unchg)
            #img_np = np.array(pil_img)
            
            ############# Simulate TORCH SERVE deployment #############
            #import handler
            #sd = handler.SlugDetect()
            #sd.handle(pil_img)
            ############## END #########################
  
            trfs = self.trans.transforms # Pointer to instantiated transform objects
            img_transformed = trfs[0](pil_img) # Resize with PIL-image -> (640,640)
            img_transformed = np.array(img_transformed) # to np-array -> (640,640)
            img_transformed = trfs[1](img_transformed) # ToTensor+scaling [0,1] -> (3,640,640)
            img_transformed = trfs[2](img_transformed) # Normalize [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            img_transformed = img_transformed.expand(1,-1,-1,-1)

            if self.device == 'cuda':
                img_transformed = img_transformed.cuda() # Convert to torch.cuda.FloatTensor

            
            y = self.online_network(img_transformed)
            a = torch.exp(y)
            #y_distance = torch.abs(y[0][0] - y[0][1])
            value, class_ind = torch.max(a,1)
            pred = 'not_slug' if class_ind[0] == 0 else 'slug'

            if filename.find("not_slug") != -1 and pred=='not_slug':
                s = '*'
                s_correct += 1
            elif filename.find("not_slug") != -1 and pred=='slug':
                s = '-'
                s_incorrect += 1
            elif filename.find("slug") != -1 and pred=='not_slug':
                s = '-'
                s_incorrect += 1
            elif filename.find("slug") != -1 and pred=='slug':
                s = '*'
                s_correct += 1

            print("Acc: {}. File: {}. Val: {}. Pred: {}.".format(s, filename, a[0].data, pred))
            i += 1
        print('Accuracy: {}. {} of {}'.format(s_correct/i, s_correct, i))
        

    def infer_from_dirs(self, img_path):

        self.online_network.eval()
        torch.inference_mode(mode=True)
        q = torch.is_inference_mode_enabled()
        #with torch.no_grad():
        
        if len(os.listdir(img_path)) == 2: # If 2 classes, SLUG, NOT_SLUG
            result = {'slug': {'correct': 0, 'not_correct': 0},
                'not_slug': {'correct': 0, 'not_correct': 0}}
        elif len(os.listdir(img_path)) == 3: # If 3 classes, EMPTY,SLUG, NOT_SLUG
            result = {'empty': {'correct': 0, 'not_correct': 0}, 
                'slug': {'correct': 0, 'not_correct': 0},
                'not_slug': {'correct': 0, 'not_correct': 0}}
        
        j = 0
        s_corr = 0
        s_incorr = 0
        for dirs in os.listdir(img_path):
            i = 0
            s_correct = 0
            s_incorrect = 0
            dir = os.path.join(img_path, dirs)
            print('************** Inferring:', dir)
            for filename in os.listdir(dir):
                #img_unchg = torchvision.io.read_image(os.path.join(dir,filename), mode=torchvision.io.ImageReadMode.UNCHANGED)
                #img_unchg = torchvision.io.decode_image(img_unchg)
                try:
                    img_unchg = torchvision.io.read_image(os.path.join(dir,filename), mode=torchvision.io.ImageReadMode.RGB) # image type = tensor
                except:
                    pass
                to_pil = torchvision.transforms.ToPILImage()
                pil_img = to_pil(img_unchg)
                #img_np = np.array(pil_img)
                
                ############# Simulate TORCH SERVE deployment #############
                #import handler
                #sd = handler.SlugDetect()
                #sd.handle(pil_img)
                ############## END #########################
    
                trfs = self.trans.transforms # Pointer to instantiated transform objects
                img_transformed = trfs[0](pil_img) # Resize with PIL-image (width,height,3) -> (640,640)
                #torchvision.transforms.functional.get_image_size(img_transformed)
                #img_transformed.save('resized.jpg') #just for test, can be removed
                img_transformed = np.array(img_transformed) # to np-array -> (640,640)
                img_transformed = trfs[1](img_transformed) # ToTensor+scaling [0,1] -> (3,640,640)
                img_transformed = trfs[2](img_transformed) # Normalize [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                img_transformed = img_transformed.expand(1,-1,-1,-1)

                if self.device == 'cuda':
                    img_transformed = img_transformed.cuda() # Convert to torch.cuda.FloatTensor

                
                y = self.online_network(img_transformed)
                a = torch.exp(y)
                #y_distance = torch.abs(y[0][0] - y[0][1])
                value, class_ind = torch.max(a,1)
                pred = 'not_slug' if class_ind[0] == 1 else 'slug' if class_ind[0] == 2 else 'empty'

                if pred == dirs:
                    s = '*'
                    s_correct += 1
                    result[dirs]['correct'] += 1
                else:
                    s = '-'
                    s_incorrect += 1
                    result[dirs]['not_correct'] += 1

                #print("Acc: {}. File: {}. Val: {}. Pred: {}.".format(s, filename, a[0].data, pred))
                i += 1
            j += i
            s_corr += s_correct
            #s_incorr += s_incorrect
            print(result)    
            print('Class: {}. Class acc: {} {} of {}'.format(dirs, s_correct/i, s_correct, i))
        print('Total accuracy: {}'.format(s_corr/j))