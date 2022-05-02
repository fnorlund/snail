# Slug data at https://data.world/datagov-uk/d27ce7af-648c-448f-824c-0e77784b09bc
# Pre-trained RESNET18 model at https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import torchvision.models as models
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages as fouoi

DATASET_DIR = './dataset'

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    dataset_dir=DATASET_DIR,
    max_samples=100,
    seed=51,
    shuffle=True,
)
dataset.persistent = True

print(dataset)

q = fouoi.get_classes(version='v6', dataset_dir='./')
openImgCls = fouoi.OpenImagesV6DatasetImporter(
    './',
    label_types=None,
    classes=None,
    attrs=None,
    image_ids=None,
    include_id=True,
    only_matching=False,
    load_hierarchy=True,
    shuffle=False,
    seed=None,
    max_samples=None
    )

classes = openImgCls.label_cls['detections']
pass

if __name__== '__main__':
    session = fo.launch_app(dataset)
    session.wait()