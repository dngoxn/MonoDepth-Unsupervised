from CNN.src.data.process_data import extract_data
import numpy as np
import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


def load_data(data_dir_path='../../data/interim/extracted_raw', data_file_name='nyu_depth_v2_labeled_extracted.npz',
              data_segmentation_amount=None):
    print('Loading data --------------------------')
    extract_data(data_dir_path, data_file_name, data_segmentation_amount)
    data_file_path = os.path.join(data_dir_path, data_file_name)
    data = np.load(data_file_path, allow_pickle=True)
    # Prepare dataset
    xs = data['images']
    ys = data['depths']
    return xs, ys


def split_train_test(xs, ys, train_test_split=0.8):
    # Shuffle the data along the first axis
    indices = np.random.permutation(len(xs))
    xs_shuffled = xs[indices]
    ys_shuffled = ys[indices]

    # Calculate split index
    split_index = int(train_test_split * len(xs_shuffled))

    # Split the data
    train_xs, test_xs = xs_shuffled[:split_index], xs_shuffled[split_index:]
    train_ys, test_ys = ys_shuffled[:split_index], ys_shuffled[split_index:]

    return train_xs, train_ys, test_xs, test_ys


def create_sample_depth_map(model, x, y):
    for i in range(len(x)):
        # Show input x
        plt.imshow(x[i].transpose(2, 1, 0))
        plt.show()
        # show model's output
        test_input = torch.from_numpy(x[i])
        model_output = model(test_input)
        model_output = torch.squeeze(model_output)
        model_output = model_output.detach().numpy().T
        plt.imshow(model_output)
        plt.title('Model Output')
        plt.show()
        # Show expected output
        plt.imshow(y[i].T)
        plt.title('Expected Output')
        plt.show()


class CNNDataset(Dataset):
    def __init__(self, xs, ys, transform=None):
        self.data = xs
        self.targets = ys
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define structure
        # Input shape: (batch_size, channels, in_height, in_width) = (batch_size, 3, 640, 480)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        # Output shape: (32, 640, 480)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # Output shape: (64, 640, 480)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # Output shape: (64, 640, 480)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        # print(x.shape)
        x = F.gelu(self.conv2(x))
        # print(x.shape)
        x = F.gelu(self.conv3(x))
        # print(x.shape)
        x = torch.squeeze(x, dim=1)
        # print(x.shape)
        return x


def test_model(model, test_loader, criterion):
    # Turn off training session (dropout, gradient calculation, etc.
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(test_loader))
        test_loss = criterion(model(inputs), targets)
        print(f'Test Loss: {test_loss:.4f}')


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1):
    print('Training model ------------------------')
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1} / {num_epochs}')
            print(f'Running loss {train_loss:.4f}')
            test_model(model, test_loader, criterion)
            # Turn back training
            model.train()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = CNNModel()

    # Hyper-parameters
    lr = 0.0005
    batch_size = 128
    max_epochs = 8
    # criterion = nn.KLDivLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # data_segmentation_amount = 10
    # xs, ys = load_data(data_file_name=f'nyu_depth_v2_labeled_extracted_{data_segmentation_amount}.npz',
    #                    data_segmentation_amount=data_segmentation_amount)
    xs, ys = load_data()
    train_xs, train_ys, test_xs, test_ys = split_train_test(xs, ys)
    print('Train data shape:', train_xs.shape)
    print('Test data shape:', test_xs.shape)
    train_DataSet = CNNDataset(train_xs, train_ys)
    train_DataLoader = DataLoader(train_DataSet, batch_size=batch_size, shuffle=True)
    test_DataSet = CNNDataset(test_xs, test_ys)
    test_DataLoader = DataLoader(test_DataSet, batch_size=len(test_xs), shuffle=True)

    train_model(model, train_DataLoader, test_DataLoader, criterion, optimizer, num_epochs=max_epochs)

    # Test output
    num_to_show = 2
    test_input = xs[:num_to_show]
    expect_output = ys[:num_to_show]
    create_sample_depth_map(model, test_input, expect_output)
