import os
import time
import numpy as np
import json
import csv
import random

import skimage
from imgaug import augmenters as iaa

from dataset import Taco
import model as modellib
from model import MaskRCNN
from config import Config
import visualize
import utils
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


def test_dataset(model, dataset, nr_images, config):

    for i in range(nr_images):

        image_id = dataset.image_ids[i] if nr_images == len(dataset.image_ids) else random.choice(dataset.image_ids)

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]

        r = model.detect([image], verbose=0)[0]

        print(r['class_ids'].shape)
        if r['class_ids'].shape[0]>0:
            r_fused = utils.fuse_instances(r)
        else:
            r_fused = r

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))

        # Display predictions
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], title="Predictions", ax=ax1)

        visualize.display_instances(image, r_fused['rois'], r_fused['masks'], r_fused['class_ids'],
                                     dataset.class_names, r_fused['scores'], title="Predictions fused", ax=ax2)

        # # Display ground truth
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, title="GT", ax=ax3)

        # Voilà
        plt.show()


#main
def main():

    # Read map of target classes
    nr_classes = 0
    class_names = []
    class_map_csv = "/home/joshlo/Downloads/TACO-master/detector/taco_config/map_1.csv"
    class_map = {}
    map_to_one_class = {}
    with open(class_map_csv) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[1]: row[0] for row in reader}
        map_to_one_class = {c: 'Litter' for c in class_map}
        class_map["BACKGROUND"] = "WANK BG!!!!!!"

    print(class_map)
    nr_classes = len(class_map)
    print(nr_classes)

    class TacoTestConfig(Config):
        NAME = "taco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        NUM_CLASSES = nr_classes
        USE_OBJECT_ZOOM = False

    ROOT_DIR = os.path.abspath("../")
    IMAGE_DIR = os.path.join(ROOT_DIR, "customdata")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "detector/models/logs")
    config = TacoTestConfig()
    config.display()
    model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    ROOT_DIR = os.path.abspath("./models")
    model_path = os.path.join(ROOT_DIR, "mask_rcnn_taco_0200.h5")
    print(model_path)

    model.load_weights(model_path, None, by_name=True)

    # load image
    file_names = next(os.walk(IMAGE_DIR))[2]
    image_name = random.choice(file_names)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
    print(IMAGE_DIR)

    # resize image
    image, window, scale, padding, crop = utils.resize_image(image, min_dim=800, max_dim=1024)

    # Run detection
    print(image)
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    print("RResults")
    print(results)
    print("RRR")
    print(r)

    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'], captions=100*["Trash"])
    # def draw_boxes(image, boxes=None, refined_boxes=None,
    #                masks=None, captions=None, visibilities=None,
    #                title="", ax=None):
    # visualize.draw_boxes(image, masks=r["masks"])
    image = np.zeros(np.shape(image))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                   class_names, r['scores'], show_bbox=False, colors=100*[(1, 1, 1)], captions=100*[""],save_location='masks/{}.jpg'.format(image_name))

if __name__ == '__main__':
    main()