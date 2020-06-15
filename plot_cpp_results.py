import os
import time
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data import FacesDB
import cv2

def _parse_file(filepath: str):
    boxes, labels, scores = [], [], []
    for line in open(filepath).readlines():
        elements = line.split(',')
        boxes.append([float(i) for i in elements[:4]])
        scores.append(float(elements[4]))
        labels.append(float(elements[5]))
    return {'boxes': boxes, 'labels': labels, 'scores': scores}

def load_dataset():
    """
    Returns the dataset"""
    path = '/home/fabian/data/TS/CrossCalibration/TCLObjectDetectionDatabase'
    path += '/greyscale.xml'
    return FacesDB(path)

def load_predictions_and_images(folder: str, dataset) -> Tuple[List[Dict]]:
    """
    parses the predicitons at path and returns a list of results
    """
    predictions, imgs = [], []
    for file in os.listdir(folder):
        prediction = _parse_file(folder + file)
        new_file = file.split('.')[0] + '.png'
        try:
            img = dataset.pull_image(new_file)
            predictions.append(prediction)
            imgs.append(img)
        except:
            continue
    return predictions, imgs

def _load_image(location: str):
    """Loads the image from the location
    """
    img = cv2.imread(location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = (img / 255).astype(np.float32)
    return img

def load_predictions_and_images_from_raw(folder: str, img_path: str) -> Tuple[List[Dict]]:
    """
    parses the predicitons at path and returns a list of results
    """
    predictions, imgs = [], []
    breakpoint()
    for file in os.listdir(folder):
        prediction = _parse_file(folder + file)
        new_file = file.split('.')[0] + '.png'
        img = _load_image(img_path + new_file)
        imgs.append(img)
        predictions.append(prediction)
    return predictions, imgs

def _add_patch(rec, axis, color):
    width, height = rec[2] - rec[0], rec[3] - rec[1]
    patch = patches.Rectangle((rec[0], rec[1]), width, height, linewidth=1,
                              edgecolor=color, facecolor='none')
    axis.add_patch(patch)

def _visualize_box(img, boxes) -> None:
    """
    Returns the list of picutres as the result
    """
    fig, axis = plt.subplots()
    axis.imshow(img)
    for rec in np.array(boxes):
        _add_patch(rec, axis, color='g')
    return fig

def visualize_prediction(predictions, images):
    """
    Plots all the predictions for one second
    """
    for img, prediction in zip(images, predictions):
        fig = _visualize_box(img, prediction['boxes'])
        plt.show(block=False)
        plt.pause(2)
        plt.close()

if __name__ == "__main__":
    FACES = load_dataset()
    RESULTS = 'cpp_client/build/results/'
    IMG_PATH = '/home/fabian/data/TS/CrossCalibration/ImageTCL/greyscale/'
    PREDICTIONS, IMGS = load_predictions_and_images_from_raw(RESULTS, IMG_PATH)
    visualize_prediction(PREDICTIONS, IMGS)
