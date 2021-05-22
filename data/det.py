import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd


class DetectionDataset(Dataset):
    def __init__(self, path, transform=None):
        self.df = pd.read_table(path, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def pull_item(self, index):
        image_path, anno = self.df.iat[index, 0], self.df.iat[index, 1]
        img = cv2.imread(image_path)
        height, width, channels = img.shape
        target = [[int(i) for i in bbox.split(',')] for bbox in anno.split(';')]
        target = [[(item[0] - 1) / width, (item[1] - 1) / height,
                   (item[2] - 1) / width, (item[3] - 1) / height,
                   item[4]] for item in target]
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        image_path = self.df.iat[index, 0]
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

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
