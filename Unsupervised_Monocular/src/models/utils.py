import os

import torch
import collections
import numpy as np
from matplotlib import pyplot as plt


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if 30 <= epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_output_numpy(output: np.ndarray, dir_path='../output/numpy', filename='output.npy'):
    ensure_dir_path(dir_path)
    save_filepath = os.path.join(dir_path, filename)
    np.save(save_filepath, output)


def read_output_numpy(dir_path='../output/numpy', filename='output.npy') -> np.ndarray:
    ensure_dir_path(dir_path)
    output = np.load(os.path.join(dir_path, filename))
    return output


def save_output_images(output: np.ndarray, dir_path='../output/images', filename='output.jpg') -> None:
    ensure_dir_path(dir_path)
    output_filepath = os.path.join(dir_path, filename)
    plt.imshow(output)
    plt.imsave(output_filepath, output)


def save_model(model: torch.nn.Module, model_id: str, dir_path='../trained_models') -> None:
    print("Saving model...")
    ensure_dir_path(dir_path)
    save_filepath = os.path.join(dir_path, model_id + '.pt')
    torch.save(model.state_dict(), save_filepath)
    print("Model saved!")


def load_model(model_id: str, dir_path='../trained_models') -> torch.nn.Module:
    print("Loading model...")
    ensure_dir_path(dir_path)
    save_filepath = os.path.join(dir_path, model_id + '.pt')
    print("Model loaded!")
    return torch.load(save_filepath)


def ensure_dir_path(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)