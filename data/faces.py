r"""
Class to the python faces
"""

from typing import List, Tuple
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


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
        img = Image.open(sample.get('file'))
        if img.mode == 'L':
            img = img.convert('RGB')
        height, width = img.height, img.width
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
        # breakpoint()
        img, target, height, width = self._load_sample(index)
        img = F.to_tensor(img)
        img = img.numpy().transpose(1, 2, 0)

        #TODO: I NEED TO ADD THIS AS I ALWAYS TRANFORMS!!!
        # if self.target_transform is not None:
            # target = self.target_transform(target, width, height)


        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            # img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # img = img.astype(float)
        # img = F.resize(img, (300, 300))
        img = F.to_tensor(img)
        return img, target, height, width
        # return torch.from_numpy(img), target, height, width

    def __getitem__(self, index):
        img, gts = self.pull_item(index)[:2]
        return img, gts

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        sample = self.ids[index]
        img = cv2.imread(sample.get('file'))
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
