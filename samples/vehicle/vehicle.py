"""
Mask R-CNN
Configurations and data loading code for MS COCO.

------------------------------------------------------------
训练模型通过网上的图片数据集，以此模型为基础再训练自己的模型
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import ipdb
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
# imgaug是一个封装好的用来进行图像augmentation的python库,支持关键点(keypoint)和bounding box一起变换
import skimage

pdb=ipdb.set_trace

import sys
mrcnn_path = '/home/ferryliu/code/CV/Mask_CRNN'
sys.path.append(mrcnn_path)
from mrcnn import visualize

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
# zipfile是python里用来做zip格式编码的压缩和解压缩的
import zipfile
import urllib.request
import shutil

import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class VehicleConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "vehicle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9 # COCO has 10 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if subset == 'val':
            dataset_dir = '/home/ferryliu/code/CV/Mask_CRNN/samples/coco/annotations/val_data.json'
        else:
            dataset_dir = "/home/ferryliu/code/CV/Mask_CRNN/samples/coco/annotations/train_data.json"
        coco = COCO(dataset_dir)
        image_dir = '/home/ferryliu/code/CV/Mask_CRNN/samples/coco/val2014'
        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap 基于位的映射[height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        # annotation 注释
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


#  -------------------------------------------------------fill colors
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

# ------------------------------------------------------------------------------------查看效果
def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """
    Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data------------------------------------什么格式
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation -----------------检测几条数据
    """

    #######################
    loop = 0
    detection_num = 0
    while True:
        img_transfer_path = '/home/ferryliu/data/Image_OLD/'
        img_source_path = '/home/ferryliu/data/Image'
        filelist = os.listdir(img_source_path)
        # sort file as it last mend time
        filelist = sorted(filelist, key=lambda x: os.path.getmtime(os.path.join(img_source_path, x)), reverse=True)
        t_prediction = 0
        t_start = time.time()
        # full name of all input iamges
        imgs_names = []
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
                src = os.path.join(os.path.abspath(img_source_path), item)
                print('item_names------------------->', item)
                if len(item) > 10:
                    imgs_names.append(src)
                else:
                    print('error file length', len(item))
                    os.remove(src)
        if limit == 0 or limit > len(imgs_names):
            limit = len(imgs_names)
        image_num = 0
        print(imgs_names)

        while image_num < limit:
            # Load image
            try:
                #  -----------------------------------------------------找到要分类的图片---------------------------------------
                imgs_per_epoch = config.BATCH_SIZE
                images = []
                for i in range(imgs_per_epoch):
                    image_scr_name = imgs_names[i+image_num]
                    try:
                        # 提取图片的numpy格式
                        image = dataset.load_image(image_scr_name)
                    except:
                        print('del error image')
                        os.remove(image_scr_name)
                    print('image length ', len(image))
                    images.append(image)

                # handle results, output a list
                res = model.detect(images, verbose=0)
                ipdb.set_trace()

                for num in range(len(res)):
                    r = res[i]
                    # image = images[i]
                    length_, width_ = image.shape[:2]
                    classfication_path = '/home/ferryliu/data/Cls_image/'
                    if not os.path.exists(classfication_path):
                        os.mkdir(classfication_path)

                    figsize = (length_, width_)
                    classfication_filename = classfication_path + os.path.basename(imgs_names[image_num + num])
                    try:
                        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names,
                                                r['scores'], show_bbox=True
                                                , show_mask=False, title='DSD 图片识别', save_path=classfication_filename)

                        print('=============================The file name of evaluation is {}'.format(os.path.basename(imgs_names[image_num + num])))
                    except:
                        shutil.move(image_scr_name, img_transfer_path + os.path.basename(image_scr_name).split('.')[0] + '.jpg')

                    # after image load, move old file to Image_OLD
                    detection_num += 1
                    # time.sleep(0.2)
                    shutil.move(image_scr_name, img_transfer_path + os.path.basename(image_scr_name).split('.')[0] + '.jpg')

                image_num = image_num + imgs_per_epoch

            except:
                image_num = image_num + 1
                print('exception no file')
                info = sys.exc_info()
                print(info[0],'----------------------------', info[1])
                print('POST error')
                # exit()

            print('img num', image_num, '-----limit ', limit)


    eval_num = 0

    ########################

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        #  help="'train' or 'evaluate' on MS COCO")
                        help="'t' or 'e' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="/home/ferryliu/code/CV/Mask_CRNN/samples/coco",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        default="/home/ferryliu/code/CV/Mask_CRNN/mask_rcnn_coco.h5",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        #  default="../../logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # -------------------------------------------------------------------------默认用500张图片来验证模型
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    # train
    if args.command == "t":
        args.command = "train"
        config = VehicleConfig()
    else:
        args.command = "evaluate"
        class InferenceConfig(VehicleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    #  # Load weights
    #  print("Loading weights ", model_path)
    #  model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        # LOAD MODEL
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        #  val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time  水平镜面翻转
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=9,
                    layers='heads',   # -----------------------------------我的一个难点 选择多个head layers to train
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=10,
                    layers='all',
                    augmentation=augmentation)

        # save model
        saved_path = '/home/ferryliu/code/CV/Mask_CRNN/samples/mrcnn_vehicle.h5'
        model.keras_model.save_weights(saved_path)

    elif args.command == "evaluate":
        # load model
        model_path = '/home/ferryliu/code/CV/Mask_CRNN/model/mrcnn_vehicle.h5'
        model.load_weights(model_path, by_name=True)

        # Validation dataset
        dataset_val = CocoDataset()
        vehicle = dataset_val.load_coco(args.dataset,'val', '2014')
        dataset_val.prepare()
        evaluate_coco(model, dataset_val,'xxx','bbox',limit=300)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
