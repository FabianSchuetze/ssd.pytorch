import numpy as np
from data import FacesDB
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def box_fits(box, xmin, ymin, xmax, ymax):
    xaxis = (box[0] > xmin) and (box[2] < xmax - xmin)
    yaxis = (box[1] > ymin) and (box[3] < ymax - ymin)
    return xaxis and yaxis

def modify_sizes(img, box, xmin, ymin, xmax, ymax):
    # breakpoint()
    box = np.array(box)
    box[:, :4] *= 300
    new_width = img.shape[1]
    new_height = img.shape[0]
    scale_width = new_width / 300
    scale_height = new_height / 300
    box[:, 0] = (box[:, 0] -  xmin)
    box[:, 1] = (box[:, 1] - ymin)
    box[:, 2] = (box[:, 2] - xmin)
    box[:, 3] = (box[:, 3] - ymin)
    return box

def adjust_target(img, boxes, xmin, ymin, xmax, ymax):
    """
    Adjusts the targets to fit the subset specified by  all the ingredients
    above
    """
    fits = []
    xscale = 300
    yscale = 300
    for box in boxes:
        if box_fits(box, xmin/xscale, ymin/yscale, xmax/xscale, ymax/yscale):
            fits.append(box)
    # breakpoint()
    if len(fits) > 0:
        return modify_sizes(img, fits, xmin, ymin, xmax, ymax)
    return []

def area(boxes):
    if type(boxes) == list:
        boxes = np.array(boxes)
    area = (boxes[:, 2] - boxes[:,0]) * ( boxes[:, 3] - boxes[:, 1])
    return area

def _add_patch(rec, axis, color):
    width, height = rec[2] - rec[0], rec[3] - rec[1]
    patch = patches.Rectangle((rec[0], rec[1]), width, height, linewidth=1,
                              edgecolor=color, facecolor='none')
    axis.add_patch(patch)

def _visualize_box(img, boxes, axis) -> None:
    """
    Returns the list of picutres as the result
    """
    # fig, axis = plt.subplots()
    axis.imshow(img)
    for rec in np.array(boxes):
        _add_patch(rec, axis, color='g')
    return fig

if __name__ == "__main__":
    path = '/home/fabian/data/TS/CrossCalibration/TCLObjectDetectionDatabase'
    path += '/greyscale.xml'
    dataset = FacesDB(path)
    np.random.seed(0)
    # for i in range(0, 20):
    img, target = dataset[20]
    xmin = np.random.randint(0, 50)
    ymin = np.random.randint(0, 50)
    width, height = np.random.randint(100, 300), np.random.randint(100, 300)
    xmax = min(300, xmin + width)
    ymax = min(300, ymin + height)
    img = np.array(img).transpose(1, 2, 0)
    crop = img[xmin:xmax, ymin: ymax, :]
    # resized = cv2.resize(crop, (300, 300))
    new_target = adjust_target(crop, target, xmin, ymin, xmax, ymax)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,  sharey=True,
                                   gridspec_kw={'height_ratios': [1, 1]})
    resized = np.ones((300, 300, 3))
    resized[xmin:xmax, ymin:ymax, :] = crop
    _visualize_box(resized, new_target, ax1)
    _visualize_box(img,  np.array(target)*300, ax2)
    # if len(new_target) > 0:
        # # x_scale = 300 / crop.shape[1]
        # # y_scale = 300 / crop.shape[0]
        # # new_target[:, 0] *= x_scale
        # # new_target[:, 1] *= y_scale
        # # new_target[:, 2] *= x_scale
        # # new_target[:, 3] *= y_scale
        # _visualize_box(crop, new_target, ax1)
    # else:
        # ax1.imshow(crop)
    # _visualize_box(img, np.array(target)*300, ax2)
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close('all')
