import torch
from torchvision.transforms import transforms, AutoAugment, AutoAugmentPolicy
from data.gaussian_blur import GaussianBlur
import numpy as np
import yaml
config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
no_classes = config['transfer_datasets']['no_classes']

def get_simclr_data_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[:2]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    return data_transforms


def transfer_transforms(input_shape, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    trans_transforms = {'train': transforms.Compose([#transforms.RandomResizedCrop(size=eval(input_shape)[:2], scale=(0.5, 1.0)),
                                        transforms.Resize(size=eval(input_shape)[:2]),
                                        transforms.RandomHorizontalFlip(p=0.2),
                                        transforms.RandomVerticalFlip(p=0.2),
                                        transforms.RandomApply([color_jitter], p=0.2),
                                        #AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                                        #transforms.RandomGrayscale(p=0.3),
                                        #transforms.GaussianBlur(kernel_size=(int(eval(input_shape)[0]//128), int(eval(input_shape)[1]//128)),sigma=(1.0, 2.0)), # for (640,640)
                                        #transforms.GaussianBlur(kernel_size=(int(eval(input_shape)[0]//546), int(eval(input_shape)[1]//410)),sigma=(1.0, 2.0)), # for (1640,1232)
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ]),
                'val': transforms.Compose([transforms.Resize(size=eval(input_shape)[:2]),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])}
    return trans_transforms

def target_transforms(x):
    pass
    #t_transforms = transforms.Compose(torch.tensor([1.,0.]) if x==1 else torch.tensor([0.,1.]))
    # slug=[1,0] not_slug=[0,1]
    #t_transforms = torch.tensor([1.,0.]) if x==1 else torch.tensor([0.,1.])
    if no_classes == 2: # index x=1->SLUG->class=1, index x=0->NOT_SLUG->class=0
        t_transforms = torch.tensor(0) if x==0 else torch.tensor(1) 
    elif no_classes == 3: # index x=0->EMPTY->class=0, index x=1->NOT_SLUG->class=1, index x=0->SLUG->class=2
        t_transforms = torch.tensor(0) if x==0 else torch.tensor(1) if x==1 else torch.tensor(2)
    return t_transforms

def infer_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=eval(input_shape)[:2]),
                                          #transforms.RandomHorizontalFlip(),
                                          #transforms.RandomApply([color_jitter], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2),
                                          #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[:2])),
                                          transforms.Resize(size=eval(input_shape)[:2]), # takes both
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          #transforms.RandomHorizontalFlip(),
                                          #transforms.CenterCrop(input_shape),
                                          #transforms.ToTensor(),
                                          #transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), # takes tensor
                                          #transforms.ToPILImage() #takes tensor or nd_array
                                          #transforms.ToTensor(), #PIL
                                          ])
    return data_transforms


def get_single_transform(name=None, input_shape=None, s=1, param1=None, param2=None):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    transform_dict = {'Resize':transforms.Resize(size=eval(input_shape)[:2]),
                        'RandomResizedCrop': transforms.RandomResizedCrop(size=eval(input_shape)[:2]),
                        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(),
                        'RandomApply': transforms.RandomApply([color_jitter], p=0.8),
                        'RandomGrayscale': transforms.RandomGrayscale(p=0.2),
                        'GaussianBlur': GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                        'ToTensor': transforms.ToTensor(),
                        'Normalize': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        }
    return transform_dict[name]


def denormalize(normalized_image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * normalized_image + mean
    return inp


def get_simclr_data_transforms_vision(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          #transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          #transforms.ToTensor()
                                          )
    scripted_transforms = torch.jit.script(data_transforms)
    return scripted_transforms