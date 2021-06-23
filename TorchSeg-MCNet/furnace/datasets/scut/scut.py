import numpy as np

from datasets.BaseDataset import BaseDataset
class Scut(BaseDataset):

    @classmethod
    def get_class_colors(*args):
        return np.array([
            [0, 0, 0],
            [128, 64, 128],
            [60, 20, 220],
            [0, 0, 255],
            [142, 0, 0],
            [70, 0, 0],
            [153, 153, 190],
            [35, 142, 107],
            [100, 60, 0],
            [153, 153, 153]
        ])

    @classmethod
    def get_class_names(*args):
        return ['background', 'road', 'person', 'rider', 'car', 'truck', 'fence', 'tree', 'bus', 'pole']

    @classmethod
    def transform_label(cls, pred, name):
        pass