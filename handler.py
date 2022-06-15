import base64
import io
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from ts.torch_handler.base_handler import BaseHandler

from preprocessing import Normalize, Rescale

class SlugDetect(BaseHandler):
# https://pytorch.org/serve/custom_service.html#custom-handler-with-class-level-entry-point
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out


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