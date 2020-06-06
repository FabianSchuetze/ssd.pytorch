r"""
An evaluation file that is compatible with how to do predicions with the cpp
client
"""
import os
from typing import List, Dict, Tuple
import numpy as np
from chainercv.evaluations import eval_detection_coco
from data import FacesDB, config
from utils.augmentations import SmallAugmentation



def _convert_box(pred_box: List[List[float]]):
    """Converts box to a form that can be used by cocoeval"""
    box = np.array(pred_box)
    return box[:, [1, 0, 3, 2]]


def convert_gt(gt):
    """
    Convert the result to a useful from
    """
    boxes = gt[:, :4]
    boxes[:, 0] *= 300
    boxes[:, 2] *= 300
    boxes[:, 1] *= 300
    boxes[:, 3] *= 300
    boxes = boxes[:, [1, 0, 3, 2]]
    label = np.array(gt[:, 4] + 1, dtype=np.int32)
    return boxes, label

def eval_boxes(predictions, gts):
    """Returns the coco evaluation metric for box detection.

    Parameters
    ----------
    predictions: List[Dict]
        The predictions. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    gts: List[Dict]
        The gts. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    Returns
    -------
    eval: Dict:
        The results according to the coco metric. At IoU=0.5: VOC metric.
    """
    assert len(predictions) == len(gts), "Preds and gts must have sam len"
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    for pred, gt in zip(predictions, gts):
        if len(pred['boxes']) > 0:
            pred_boxes.append(_convert_box(pred['boxes']))
            pred_labels.append(np.array(pred['labels'], dtype=np.int32))
            pred_scores.append(np.array(pred['scores']))
            gt_box, gt_label = convert_gt(gt)
            gt_boxes.append(gt_box)
            gt_labels.append(gt_label)
    res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                              gt_boxes, gt_labels)
    return res

def _parse_file(filepath: str):
    boxes, labels, scores = [], [], []
    for line in open(filepath).readlines():
        elements = line.split(',')
        boxes.append([float(i) for i in elements[:4]])
        scores.append(float(elements[4]))
        labels.append(float(elements[5]))
    return {'boxes': boxes, 'labels': labels, 'scores': scores}


def load_predictions_and_gts(folder: str, dataset) -> Tuple[List[Dict]]:
    """
    parses the predicitons at path and returns a list of results
    """
    predictions = []
    gts = []
    for file in os.listdir(folder):
        prediction = _parse_file(folder + file)
        new_file = file.split('.')[0] + '.png'
        try:
            gt = dataset.pull_anno(new_file)
            predictions.append(prediction)
            gts.append(gt)
        except:
            continue
    return predictions, gts

def load_dataset():
    """
    Returns the dataset"""
    cfg = config.faces
    path = '/home/fabian/data/TS/CrossCalibration/TCLObjectDetectionDatabase'
    path += '/greyscale_combined.xml'
    return FacesDB(path, transform=SmallAugmentation(cfg['min_dim'],
                                                     config.MEANS))

if __name__ == "__main__":
    FACES = load_dataset()
    PREDICTIONS, GTS = load_predictions_and_gts('cpp_client/build/results/',
                                                FACES)
    RES = eval_boxes(PREDICTIONS, GTS)
    print(RES['coco_eval'].__str__())
