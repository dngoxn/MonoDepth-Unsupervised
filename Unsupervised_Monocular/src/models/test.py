from torch.utils.data import DataLoader
from Unsupervised_Monocular.src.data.data_loader import KittiDataSet
from Unsupervised_Monocular.src.data.transforms import get_transforms
from Unsupervised_Monocular.src.models.models import Resnet50MonoDepth
from Unsupervised_Monocular.src.models.loss import MonoDepthLoss
from Unsupervised_Monocular.src.models.utils import *
from matplotlib import pyplot as plt
import numpy as np


LEFT_IMG_KEY = 'left_img'
RIGHT_IMG_KEY = 'right_img'
root_dir_right = '../../data/kitti/data_object_image_3/training'
root_dir_left = '../../data/kitti/data_object_image_2/training'
train_transforms = get_transforms(size=(256, 512), mode='train')
kitti_train_dataset = KittiDataSet(root_dir_left, root_dir_right, 'train', train_transforms)
kitti_train_dataloader = DataLoader(kitti_train_dataset, batch_size=8, shuffle=False)
test_dir_left = '../../data/kitti/data_object_image_2/testing'
test_dir_right = '../../data/kitti/data_object_image_2/testing'
test_transforms = get_transforms(size=(256, 512), mode='test')
kitti_test_dataset = KittiDataSet(test_dir_left, test_dir_right, 'test', test_transforms)
kitti_test_dataloader = DataLoader(kitti_test_dataset, batch_size=8, shuffle=False)

model = Resnet50MonoDepth(3)


def test_loader():
    for data in kitti_train_dataloader:
        left = data[LEFT_IMG_KEY]
        right = data[RIGHT_IMG_KEY]
        right = data[RIGHT_IMG_KEY]
        print(left.dtype, right.dtype)
        print(left.shape, right.shape)

        # show transformed images
        img_left = left[0].detach().numpy().astype(np.float32).transpose((1, 2, 0))
        img_right = right[0].detach().numpy().astype(np.float32).transpose((1, 2, 0))
        print(img_left.shape, img_right.shape)
        plt.imshow(img_left)
        plt.show()
        plt.imshow(img_right)
        plt.show()
        break


def test_model():
    loss_fn = MonoDepthLoss()
    for data in kitti_train_dataloader:
        left = data[LEFT_IMG_KEY]
        right = data[RIGHT_IMG_KEY]
        disp = model(left)
        loss = loss_fn(disp, (left, right))
        print(loss)

        output = disp[0][0, 0, :, :]
        output = output.detach().numpy().astype(np.float32)
        plt.imshow(output)
        plt.show()
        break


def test_save_output_numpy():
    for i, data in zip(range(3), kitti_test_dataloader):
        disp = model(data)
        outputs = disp[0][:, 0, :, :]
        outputs = outputs.detach().numpy().astype(np.float32)
        print(outputs.shape)
        save_output_numpy(outputs, filename=f'test_output_{i}')


def test_save_output_images():
    for i, data in zip(range(3), kitti_test_dataloader):
        disp = model(data)
        outputs = disp[0][:, 0, :, :]
        outputs = outputs.detach().numpy().astype(np.float32)
        save_output_numpy(outputs, filename=f'test_output_{i}')
        for j, output in enumerate(outputs):
            save_output_images(output, filename=f'test_output_{i}_{j}.png')


def test_save_model():
    save_model(model, 'test_model')


def test_load_model():
    return load_model('test_model')


if __name__ == '__main__':
    # test_loader()
    # test_model()
    # test_save_output_numpy()
    # print('Saving output images...')
    # test_save_output_images()

    test_save_model()
    loaded_model = test_load_model()
    model.load_state_dict(loaded_model)
    input = torch.rand(8, 3, 256, 512)
    outputs = model(input)
    for output in outputs:
        print(output.size())
