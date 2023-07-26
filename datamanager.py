# https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
# https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=normalize#torchvision.transforms.Normalize
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning
# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices

# https://pytorch.org/vision/stable/models.html
# Instancing a pre-trained model will download its weights
# to a cache directory. This directory can be set using the
# TORCH_HOME environment variable. See 
# https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url
# for details.

from __future__ import print_function, division
#from typing import Concatenate
from sklearn.semi_supervised import LabelSpreading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets #,transforms
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data.transforms import transfer_transforms, infer_transforms, target_transforms
from torch.utils.tensorboard import SummaryWriter

print('Torchvision version:',torchvision.__version__)

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs')

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data transforms from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

class SlugDataset():
    def __init__(self, data_dir, *args, **kwargs):
        super(SlugDataset, self).__init__()
        self.kwargs = kwargs
        self.input_shape = self.kwargs['transforms']['input_shape']
        self.no_classes = self.kwargs['no_classes']
        self.data_transforms = transfer_transforms(
            self.kwargs['transforms']['input_shape'], 
            self.kwargs['transforms']['s'])
        self.q_transforms = infer_transforms(
            self.kwargs['transforms']['input_shape'], 
            self.kwargs['transforms']['s'])

        self.target_transforms = target_transforms
        
        #fahrenheit = {'t1':0, 't2':10, 't3':20, 't4':30}
        #celsius = {k:(float(5)/9)*(v-32) for (k,v) in fahrenheit.items()}
        #print(celsius)

        self.data_dir = data_dir
        self.img_size = self.kwargs['transforms']['input_shape'] 
       
        self.image_dataset1 = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), transform=self.data_transforms[x], 
                                target_transform=self.target_transforms) for x in ['train', 'val']} # data_transforms
        '''
        self.image_dataset2 = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), transform=self.q_transforms, 
                                target_transform=self.target_transforms) for x in ['train', 'val']} # data_transforms
        '''
        
        self.image_datasets = self.image_dataset1
        self.image_train_datasets = ConcatDataset([self.image_dataset1['train']])
        self.image_val_datasets = ConcatDataset([self.image_dataset1['val']])
        pass
        # self.image_datasets.datasets[0]{'train} etc....
        
        #self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=1,shuffle=True) for x in ['train', 'val']}
        
        #self.write_img_to_tensorboard()
    
    def __len__(self):
        #return len(self.image_train_datasets)
        return len(self.image_datasets)

    def __getitem__(self, idx):
        #return self.image_train_datasets[idx]
        return self.image_datasets[idx]
    
    def get_no_images(self, type=['train', 'val']):
        # NOT READY
        i = 0
        for key in self.image_datasets.keys():
            for _ in self.image_datasets[key]:
                i += 1
        return i
        
    def imshow(self, inp, title=None):
        # NOT READY
        iter = iter(self.dataloaders[type])
        image, target = iter.next()
        pass
        
        # plot image
        """Imshow for Tensor."""
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning
        
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

        #q=eval(self.kwargs['transforms']['input_shape'])
        pass

    def plot(self, imgs, with_orig=True, row_title=None, **imshow_kwargs):
        # https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        # NOT READY
        if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        # orig_img = Image.open(Path('assets') / 'astronaut.jpg') # Ex how to call function
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [orig_img] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()

    def write_img_to_tensorboard(self, img=None, path=None):
        import numpy as np
        #q_transforms = self.data_transforms
        q_transforms = self.q_transforms
        
        # view images using q_transforms
        q_image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), q_transforms) for x in ['train', 'val']} # data_transforms
        q_dataloaders = {x: torch.utils.data.DataLoader(q_image_datasets[x], batch_size=1,
                                                    shuffle=True, num_workers=1)
                    for x in ['train', 'val']}
        
        # get some random training images
        img_slug = []
        img_not_slug = []
        dataiter = iter(q_dataloaders['val'].dataset.imgs)
        for img_path, label in dataiter:
            img_unchg = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.UNCHANGED)
            img_tensor = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # image type = tensor
            # ToPilImage info https://chowdera.com/2022/02/202202160612032377.html
            to_pil = transforms.ToPILImage() 
            pil_img = to_pil(img_unchg)

            img_np = np.array(pil_img)

            trfs = q_transforms.transforms # Pointer to instantiated transform objects
            img_transformed = trfs[0](pil_img) # Resize with PIL-image -> (640,640)
            img_transformed = np.array(img_transformed) # (640,640)
            img_transformed = trfs[1](img_transformed) # ToTensor -> (3,640,640)
            img_transformed = trfs[2](img_transformed)
            #img_transformed = q_transforms(self.img_size)(img_unchg)
            if label == 0:
                img_not_slug.append(img_transformed)
            elif label == 1:
                img_slug.append(img_transformed)
                
        # create grid of images
        img_slug_grid = torchvision.utils.make_grid(img_slug, 3)
        img_not_slug_grid = torchvision.utils.make_grid(img_not_slug, 3)

        # show images
        #plt_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('Slugs', img_slug_grid)
        writer.add_image('Not Slugs', img_not_slug_grid)