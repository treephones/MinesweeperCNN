import numpy as np

import tensorflow
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.board import create_data
import time


def write_boards(boards, filename):
    np.savez_compressed(filename, *boards)
    print(f'Wrote dataset of size {len(boards)} to {filename}')

def load_boards(filename):
    with np.load(filename) as data:
        print(f'Loading dataset of size {len(data)} from {filename}')
        return [data[key] for key in data]

def create_dataset(size):
    data = create_data(30, 16, 99, size)
    dataset = [point[0] for point in data]
    print('-------------------------------------------------------------')
    labels = [point[1] for point in data]
    coverage_mask = [point[2] for point in data]
    write_boards(dataset, 'board_dataset.npz')
    write_boards(labels, 'labels.npz')
    #write_boards(coverage_mask, 'coverage_mask.npz')

def get_data_and_labels(xfile, yfile, zfile, coverage_mask_bool=False):
    data = load_boards(xfile)
    labels = load_boards(yfile)
    #coverage_mask = load_boards(zfile)

    #combine data and coverage_mask into shape (x,y,2)
    # if coverage_mask_bool:
    #     data = [np.concatenate((d, c), axis=-1) for d, c in zip(data, coverage_mask)]
    print(f'size {data[0].shape}')

    data = tensorflow.convert_to_tensor(data, dtype=tensorflow.float32)
    labels = tensorflow.convert_to_tensor(labels, dtype=tensorflow.float32)
    return [data, labels]

if __name__ == '__main__':
    # Start the timer
    start_time = time.time()

    create_dataset(300000)
    #data, labels = get_data_and_labels('board_dataset.npz', 'labels.npz', 'coverage_mask.npz')
    
    # Stop the timer
    end_time = time.time()
    
    print(f"Time: {end_time-start_time}")
    # print(labels)

