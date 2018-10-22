import os
import sys
import ipdb
pdb = ipdb.set_trace
import time
import numpy as np
import imgaug
import skimage

MAIN_PATH = os.path.abspath('../..')
sys.path.append(MAIN_PATH)
from mrcnn import visualize
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASIC_MODEL_PATH = os.path.join(MAIN_PATH, "mask_rcnn_coco.h5")
# dirctory to save logs and model checkpoints, if not provided Thought the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(MAIN_PATH, 'logs')


######################################################### Configuration
class VehicleConfig(Config):
    '''
    Derives from the base Config class and overrides values specific to the vehicle dataset
    '''
    NAME = 'vehicle'
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 9

######################################################## Dataset
class VehicleDataset(utils.Dataset):
    def load_vehicle(self, dataset_dir, subset, image_dir=None):
        '''
        Load a subset of the Vehicle dataset
        dataset_dir: the root directory of the Vehicle dataset annotation
        subset: what to  load (train val)
        image_dir: the save location of images
        '''
        assert subset in ['train', 'val'], 'error subset, must train or val'
        if subset == 'train':
            dataset = os.path.join(dataset_dir,'train_data.json')
        else:
            dataset = os.path.join(dataset_dir, 'val_data.json')

        # constructor of COCO helper class for reading and visualzing annotations; creat Index
        vehicle  = COCO(dataset)
        class_ids = sorted(vehicle.getCatIds())
        # annotation中的image_ids映射到image_info中的id
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(vehicle.getImgIds(catIds=[id])))
        #remove duplicates
        image_id = list(set(images_ids))
        # or image_id = list(vehicle.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class('vehicle', i, vehicle.loadCats(i)[0]['name'])

        # Add images
        if image_dir:
            image_dir = os.path.join(MAIN_PATH, image_dir)
        else:
            image_dir = '/home/ferryliu/code/CV/Mask_CRNN/samples/coco/val2014'
        for i in image_ids:
            self.add_image('vehicle',i,
                        path=os.patimage_id.join(image_dir, vehicle.imgs[i]['file_name']),
                        width=vehicle.imgs[i]['width'],
                        height=vehicle.imgs[i]['height'],
                        #  annotations=vehicle.loadAnns(vehicle.getAnnIds(imgIds=[i], catIds=class_ids, catIds=None ))
                        annotations=vehicle.loadAnns(vehicle.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
                        )

########################################### Train
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='train mask rcnn on vehicle dataset')
    parser.add_argument('--dataset', required=False)
    parser.add_argument('--model',required=False,
                        default='last',
                        help='last or coco')
    parser.add_argument('--logs',required=False,
                    default=DEFAULT_LOGS_DIR)
    args  = parser.parse_args()

    config = VehicleConfig()
    config.display()

    # create model
    pdb()
    model = modellib.MaskRCNN(mode='training',config=config, model_dir=args.logs)

    if args.model.lower() == 'last':
        model_path = model.find_last()
        # Load MODEL
        print('loading weight ', model_path)
        model.load_weights(model_path, by_name=True)
    else:
        model_path = BASIC_MODEL_PATH
        # Load coco MODEL
        print('loading weight ', model_path)
        model.load_weights(model_path, by_name=True, exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])

    # set train dataset and validation dataset
    dataset_train =VehicleDataset()
    dataset_train.load_vehicle(args.dataset, 'train')
    dataset_train.prepare()

    datasel_val =VehicleDataset()
    datasel_val.load_vehicle(args.dataset, 'train')
    datasel_val.prepare()

    #Image augmentation right/left flip 0.5
    augmentation = imgaug.augmenters.Fliplr(0.50)

    ############# TRAINING  STAGE
    # training stage 1
    print('trainng network heads')
    model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,
            epochs=10,layers='heads', augmentation=augmentation)

    # training stage 2
    print('Fine tune resnet stage 4 and up')
    model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,
            epochs=5,layers='4+', augmentation=augmentation)

    # training stage 3
    print('trainng network heads')
    model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE/10,
           epochs=3,layers='all', augmentation=augmentation)

    # save model
    saved_path = '/home/ferryliu/code/CV/Mask_CRNN/model/mrcnn_vehicle_new.h5'
    model.keras_model.save_weights(saved_path)
    ipdb.set_trace()

