from torch.utils.data import DataLoader
from Unsupervised_Monocular.src.models.utils import *
from Unsupervised_Monocular.src.data.data_loader import KittiDataSet
from Unsupervised_Monocular.src.data.transforms import get_transforms
from Unsupervised_Monocular.src.models.loss import MonoDepthLoss
from Unsupervised_Monocular.src.models.models import Resnet50MonoDepth


LEFT_IMG_KEY = 'left_img'
RIGHT_IMG_KEY = 'right_img'


class MonoDepthModel:
    def __init__(self,
                 model,
                 model_id,
                 train_dir_left='../../data/kitti/data_object_image_2/training',
                 train_dir_right='../../data/kitti/data_object_image_3/training',
                 test_dir_left='../../data/kitti/data_object_image_2/testing',
                 test_dir_right='../../data/kitti/data_object_image_2/testing',
                 output_dir_path='../../output',
                 mode='train',
                 test=False,
                 lr=1e-4,
                 control_lr=True,
                 batch_size=8,
                 max_epochs=10):
        """
        Initialize the MonoDepth model.
        """""
        self.train_dir_left = train_dir_left
        self.train_dir_right = train_dir_right
        self.test_dir_left = test_dir_left
        self.test_dir_right = test_dir_right
        self.output_dir_path = output_dir_path
        self.mode = mode
        self.test = test
        self.lr = lr
        self.control_lr = control_lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.model = model
        self.model_id = model_id
        self.criterion = MonoDepthLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_transforms = get_transforms(mode=mode)
        self.kitti_train_dataset = KittiDataSet(self.train_dir_left, self.train_dir_right, self.mode, self.train_transforms)
        self.kitti_train_dataloader = DataLoader(self.kitti_train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_transforms = get_transforms(mode='test ')
        self.kitti_test_dataset = KittiDataSet(self.test_dir_left, self.test_dir_right, 'test', self.test_transforms)
        self.kitti_test_dataloader = DataLoader(self.kitti_test_dataset, batch_size=self.batch_size, shuffle=True)

        if self.test:
            train_dir_left = test_dir_left
            train_dir_right = test_dir_right
            mode = 'test'
            self.kitti_test_dataset = KittiDataSet(train_dir_left, train_dir_right, mode)

        # self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        # self.model.to(self.device)

    def train(self):
        if self.mode != 'train':
            print("Incorrect mode!")
            print("Set mode to 'train' to train the model")
            exit()
        print('-----------------')
        print('Start training...')
        for epoch in range(self.max_epochs):
            train_loss = 0.0
            if self.control_lr:
                adjust_learning_rate(self.optimizer, epoch, self.lr)
            disp = None
            left = None
            for data in self.kitti_train_dataloader:
                # Reset gradient
                self.optimizer.zero_grad()
                # data = to_device(data, self.device)
                left = data[LEFT_IMG_KEY]
                right = data[RIGHT_IMG_KEY]
                disp = self.model(left)
                loss = self.criterion(disp, (left, right))
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % 1 == 0:
                print(f'Epoch: {epoch} Train Loss: {train_loss:.4f}')

            if 1:
                from matplotlib import pyplot as plt
                import numpy as np
                left = left[0]
                left = left.detach().numpy().astype(np.float32).transpose((1, 2, 0))
                plt.imshow(left)
                plt.show()
                output = disp[0][0, 0, :, :]
                output = output.detach().numpy().astype(np.float32)
                plt.imshow(output)
                plt.show()

        save_model(self.model, self.model_id)

    def test(self):
        self.model = load_model(self.model_id)
        self.model.eval()
        with torch.no_grad():
            for _, data in zip(range(5), self.kitti_test_dataloader):
                # save output depth map
                pass

        self.model.train()
        return 1


if __name__ == '__main__':
    in_channels = 3
    model = MonoDepthModel(
                model=Resnet50MonoDepth(in_channels),
                model_id='model_01',
                train_dir_left='../../data/kitti/data_object_image_2/training',
                train_dir_right='../../data/kitti/data_object_image_3/training',
                mode='train',
                test=False,
                lr=1e-3,
                control_lr=True,
                batch_size=8,
                max_epochs=50
    )
    model.train()
