#import base64
import io
import os
from nbformat import write
#import time
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from ts.torch_handler.base_handler import BaseHandler
import base64

#from preprocessing import Normalize, Rescale

#from data.transforms import infer_transforms

class SlugDetect(BaseHandler):
# https://pytorch.org/serve/custom_service.html#custom-handler-with-class-level-entry-point
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.manifest = None
        #self.input_shape = ("640,640,3") # (width, height, channels)
        #self.input_shape = ("1640,1232,3")
        #self.input_shape = ("640,480,3")
        #self.input_shape = ("224,224,3")

        #self.input_shape = ("640,640,3") # (height, width, channels)
        #self.input_shape = ("1232,1640,3")
        #self.input_shape = ("480,640,3")
        self.input_shape = ("224,224,3")
        self.s = 1
        self.trans = self.infer_transforms(self.input_shape, self.s)
        print('###### CONSTRUCTOR INITIALIZED ######')
    
    def initialize(self, context): #initialize(self, context)
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """
        print('###### CUSTOM INITIALIZATION ######')
        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #print('### DEVICE=',self.device)

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        try:
            self.model = torch.jit.load(model_pt_path)
            print('##### MODEL SUCCESSFULLY LOADED #######')
        except:
            print('##### ERROR: MODEL LOAD #######')

        # Load 
        self.initialized = True

    def load_images(self, data):
        #images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # the image is sent as bytesarray. This is used!!
            image = Image.open(io.BytesIO(image))
            #images.append(image)
        
        return image
    
    def infer_transforms(self, input_shape, s=1):
        # NOTE: This is COPIED from file data.transforms.py
        # get a set of data augmentation transformations as described in the SimCLR paper.
        #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=eval(input_shape)[:2]),
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.RandomApply([color_jitter], p=0.8),
                                            #transforms.RandomGrayscale(p=0.2),
                                            #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[:2])),
                                            #transforms.ToPILImage() #takes tensor or nd_array
                                            transforms.Resize(size=eval(input_shape)[:2]), # takes both
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            #transforms.RandomHorizontalFlip(),
                                            #transforms.CenterCrop(input_shape),
                                            #transforms.ToTensor(),
                                            #transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), # takes tensor
                                            #transforms.ToTensor(), #PIL
                                            ])
        return data_transforms
        
    def _save_to_file(self, data, f_name):
        name = os.path.join('/home/fredrik/source/snail/logs/', f_name)
        try:
            #img=Image.open(data)
            img=Image.save(name,data)
        except:
            img=open(name,'wb')
            img.write(data)
            img.close()
        return None

    def _load_from_file(self):
        #img=Image.open(os.path.join('/home/fredrik/source/snail/logs/', 'q.jpg'),'r')
        #img.load()
        img=open(os.path.join('/home/fredrik/source/snail/logs/', 'q.jpg'), mode='rb')
        img.read()
        #img.close()
        return img
    
    def preprocess(self, pil_image):

        trfs = self.trans.transforms
        print('###### PIL.WIDTH',pil_image.width)
        img_transformed = trfs[0](pil_image) # Resize with PIL-image -> (height,width)
        print('###### ',img_transformed)

        # Debug
        img_transformed.save(os.path.join('/home/fredrik/source/snail/logs/', 'aaa.jpg'))
        # End debug

        img_transformed = np.array(img_transformed) # to np-array -> (width,height)
        print('###### Resized',img_transformed)
        img_transformed = trfs[1](img_transformed) # ToTensor+scaling [0,1] -> (3,width,height)
        print('###### ',img_transformed)
        img_transformed = trfs[2](img_transformed) # Normalize
        print('###### ',img_transformed)
        img_transformed = img_transformed.expand(1,-1,-1,-1)
        print('###### ',img_transformed.shape)
        return img_transformed
    
    def postprocess(self, inference_output):
        #return super().postprocess(data)
        # Convert from torch sensor to Python list?
        pred = torch.exp(inference_output).tolist()
        #pred = str(pred[0]) + ',' + str(pred[1])
        print('#### INFERENCE_OUTPUT ######', pred)
        return pred
    
    def inference(self, model_input):
        return self.model(model_input)
    
    def handle(self, image, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        #image = self._load_from_file()
        
        print('#### IMAGE ####')
        #self.context = context
        print('##### MANIFEST: ',context.manifest)
        #image = self.load_images(image)[0]
        image = self.load_images(image)
        preprocessed_image = self.preprocess(image)
        print('###### IMAGE____2')
        
        # Infer & postprocess
        pred_out = self.inference(preprocessed_image)
        classification = self.postprocess(pred_out)
        return classification


'''
class U2Net(BaseHandler):

    image_processing = Compose(
        [
            Rescale(320),
            Normalize(),
        ]
    )

    def _norm_pred(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def basic_cutout(self, img, mask):
        u2net_mask = Image.fromarray(mask).resize(img.size, Image.LANCZOS)
        mask = np.array(u2net_mask.convert("L")) / 255.0

        result = img.copy().convert("RGBA")
        return result.putalpha(mask)

    def postprocess(self, image, output):
        pred = output[0][:, 0, :, :]
        predict = self._norm_pred(pred)
        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        mask = (predict_np * 255).astype(np.uint8)

        return [self.basic_cutout(image, mask).tobytes()]

    def load_images(self, data):
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # the image is sent as bytesarray
            image = Image.open(io.BytesIO(image))
            images.append(image)

        return images

    def handle(self, data, context):
        """Entry point for handler. Usually takes the data from the input request and
           returns the predicted outcome for the input.
           We change that by adding a new step to the postprocess function to already
           return the cutout.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns the data input with the cutout applied.
        """
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        images = self.load_images(data)
        data_preprocess = self.preprocess(images)

        if not self._is_explain():
            output = self.inference(data_preprocess)
            output = self.postprocess(images, output)
        else:
            output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output
'''