import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

DATA_FOLDER_PATH = '../../data/raw'
DATA_FILE_NAME = 'nyu_depth_v2_labeled.mat'
DATA_FILE_PATH = os.path.join(DATA_FOLDER_PATH, DATA_FILE_NAME)


def print_hdf5_structure(g, indent=0):
    for i, key in enumerate(g):
        if i == 20:
            break
        if isinstance(g[key], h5py.Group):
            print(" " * indent + f"Group: {key}")
            print_hdf5_structure(g[key], indent + 4)
        else:
            print(" " * indent + f"Dataset: {key}")


def save_samples(file_name, index, dir_path='../../data/interim/samples', figure=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    samples_folder_path = os.path.join(dir_path, file_name)
    if not os.path.exists(samples_folder_path):
        os.makedirs(samples_folder_path)
    samples_file_path = os.path.join(samples_folder_path, file_name + f'_{index}.png')
    plt.imshow(figure)
    plt.savefig(samples_file_path, bbox_inches='tight')
    print(f"Saved {samples_file_path}")


def save_data(data, dir_path='../../data/interim/extracted_raw',
              extracted_file_name='nyu_depth_v2_labeled_extracted.npz'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    extracted_file_path = os.path.join(dir_path, extracted_file_name)
    np.savez(extracted_file_path, **data)  # ** for unpacking into dictionary before saving
    print(f"Saved extracted file at: {extracted_file_path}")


def create_samples():
    with h5py.File(DATA_FILE_PATH, 'r') as f:
        samplers = ['depths', 'images', 'labels', 'rawDepths']
        sampling_amounts = 5    # number of samples to be saved
        for sample_type in samplers:
            for index, sample in enumerate(f[sample_type]):
                if index >= sampling_amounts:
                    break
                # Alter shape for better visual
                if sample_type == samplers[1]:
                    save_samples(sample_type, index, figure=sample.transpose(2, 1, 0))
                else:
                    save_samples(sample_type, index, figure=sample.T)


def extract_data(dir_path='../../data/interim/extracted_raw', data_file_name='nyu_depth_v2_labeled_extracted.npz',
                 data_segmentation_amount=None):
    if not os.path.isfile(os.path.join(dir_path, data_file_name)):
        print('Extracting data -----------------------')
        with h5py.File(DATA_FILE_PATH, 'r') as f:
            data_types = ['depths', 'images', 'labels', 'rawDepths']
            # Initialize dictionary to store extracted data
            extracted_data = {data_type: [] for data_type in data_types}
            for data_type in data_types:
                data = []
                for index, im in enumerate(f[data_type]):
                    # Note: torch's expectation for cnn shape:
                    # (batch_size, num_input_channels, image_height, image_width)
                    if data_segmentation_amount and index >= data_segmentation_amount:
                        break
                    if data_type == data_types[1]:  # If data_type is `images`
                        data.append(im.astype(np.float32) / 255.0)
                    else:
                        data.append(im.astype(np.float32))
                extracted_data[data_type] = data
            save_data(extracted_data, dir_path=dir_path, extracted_file_name=data_file_name)
    else:
        print(f"Loading data from existing file {data_file_name}")


if __name__ == '__main__':
    # print('Running create_samples ----------------')
    # create_samples()
    print('Running extract_data ------------------')
    extract_data()
