import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from Unsupervised_Monocular.src.data.transforms import get_transforms


# KITTI dataset defaults images from left and right view as followed
LEFT_FILE_NAME = 'image_2'
RIGHT_FILE_NAME = 'image_3'


class KittiDataSet(Dataset):
    """Kitti Dataset"""
    def __init__(self, root_dir_left: str, root_dir_right: str, mode, transform=None):
        left_dir = os.path.join(root_dir_left, LEFT_FILE_NAME)
        self.left_path = sorted([os.path.join(left_dir, file_name) for file_name in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir_right, RIGHT_FILE_NAME)
            self.right_path = sorted([os.path.join(right_dir, file_name) for file_name in os.listdir(right_dir)])
            assert len(self.left_path) == len(self.right_path)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_path)

    def __getitem__(self, idx):
        left_img = self.load_img(self.left_path[idx])
        if self.mode == 'train':
            right_img = self.load_img(self.right_path[idx])
            sample = {'left_img': left_img, 'right_img': right_img}
            if self.transform:
                sample = self.transform(sample)
            return sample
        else:
            if self.transform:
                left_img = self.transform(left_img)
            return left_img

    def load_img(self, img_path):
        img = Image.open(img_path)
        return img


if __name__ == '__main__':
    test_dir_left = '../../data/kitti/data_object_image_2/testing'
    test_dir_right = '../../data/kitti/data_object_image_2/testing'
    mode = 'test'
    transforms = get_transforms(mode=mode)
    testDataset = KittiDataSet(root_dir_left=test_dir_left,
                               root_dir_right=test_dir_right,
                               mode=mode, transform=transforms)
    testLoader = torch.utils.data.DataLoader(testDataset)
    for i, sample in zip(range(5), testLoader):
        print(sample.shape)
