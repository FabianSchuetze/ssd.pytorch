r"""
Class to the python faces
"""

from typing import List, Tuple
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .conversion_functions import Cropping, visualize_box


class FacesDB(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, database: str, transform=None):
                 # target_transform=VOCAnnotationTransform()):
        self._database = database
        self.ids = self._load_images()
        self._conversion = {'glabella': 0, 'left_eye':1, 'right_eye':2,
                            'nose_tip': 3}
        self.transform = transform
        self._Crop = Cropping()
        # self.target_transform = target_transform
        self.name = 'Faces'
        self._filepath_storage = self._make_key_location_pair()

    def _load_images(self):
        tree = ET.parse(self._database)
        return tree.findall('images/image')

    def _make_key_location_pair(self):
        """
        Specifies the filename as index and the index in self.ids as value
        """
        storage = {}
        for idx, val in enumerate(self.ids):
            filename = val.get('file').rsplit('/')[-1]
            storage[filename] = idx
        return storage

    def _convert_to_box(self, box: ET.Element) -> List[int]:
        """
        Generates the bouding boxes
        """
        xmin = int(box.get('left'))
        ymin = int(box.get('top'))
        xmax = int(box.get('left')) + int(box.get('width'))
        ymax = int(box.get('top')) + int(box.get('height'))
        return [xmin, ymin, xmax, ymax]

    def _append_label(self, box: ET.Element) -> int:
        """
        Gets the corresponding label to the box
        """
        label = box.find('label').text
        return self._conversion[label]

    def _load_sample(self, idx) -> Tuple[List]:
        sample = self.ids[idx]
        img = cv2.imread(sample.get('file'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        res = []
        for tag in sample.findall('box'):
            sample = []
            box = self._convert_to_box(tag)
            for i in range(4):
                scale = width if i %2 == 0 else height
                sample.append(box[i] / scale)
            sample.append(self._append_label(tag))
            res += [sample]
        return img, res, height, width

    def pull_item(self, index: int):
        img, target, height, width = self._load_sample(index)
        img = cv2.resize(img, (300, 300))
        img = (img / 255).astype(np.float32)
        # if np.random.rand() < 0.3:
            # while True:
                # try:
                    # img, target = self._Crop.resize(img, target)
                    # break
                # except IndexError:
                    # pass
        #TODO: I NEED TO ADD THIS AS I ALWAYS TRANFORMS!!!
        # if self.target_transform is not None:
            # target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            boxes, labels = target[:, :4], target[:, 4]
            # img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # breakpoint()
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, target, height, width

    def __getitem__(self, index):
        img, gts = self.pull_item(index)[:2]
        return img, gts

    def __len__(self):
        return len(self.ids)

    def pull_image(self, filename: str):
        """
        Returns the original image as numpy array

        Paramters
        --------
        index: int
            The location of the image in the database

        Returns
        -------
        img: np.array
            The image
        """
        index = self._filepath_storage[filename]
        sample = self.ids[index]
        img = cv2.imread(sample.get('file'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        img = (img / 255).astype(np.float32)
        return img

    def pull_anno(self, filename: str):
        """
        Returns the annotation of the image. In contrast to the other images,
        this function takes a string as an argument which corresponsed to the
        filename
        """
        img_id = self._filepath_storage[filename]
        return self.pull_item(img_id)[1]

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
