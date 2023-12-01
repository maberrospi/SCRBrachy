from Train import load_data_npz
import numpy as np


def find_min_max(train_data):
    """Find the min and max values from all the CT slices in the training set"""
    np_list = []

    for item in train_data:
        np_list.append(item[1])
    if np_list:
        hmin = np.min(np_list)
        hmax = np.max(np_list)
        print(f'images min: {hmin}\nimages max: {hmax}')
        return hmin, hmax
    else:
        print("The list provided is empty")
        return None, None


def main():
    MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/MasksV2/maskTrain.npz'
    CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimagesV2/ctTrain.npz'
    # Load the data
    # cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
    # Train-80%,Val-10%-Test-10%
    train_cts, train_masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
    find_min_max(train_cts)


if __name__ == "__main__":
    main()
