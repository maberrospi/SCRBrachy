#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:55:26 2023

@author: ERASMUSMC+099035
"""
# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
# from tqdm import tqdm
import numpy as np
import imageio.v3 as iio
import glob, os
import random
import math


# %% DEFINE ALL FUNCTIONS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def save_CT_to_npz(CT_DIR, filename):
    all_cts = glob.glob(CT_DIR + "/*.png")
    # If you would like to sort per CT 
    # all_cts = sorted(all_cts, key = lambda f: int(''.join(filter(str.isdigit,f[:f.find("_")]) or -1)))
    sl_np_list = []
    name_list = []
    print("Saving all CT arrays to NPZ")
    # Maybe use the digits from the filenames and not the idx
    for idx, file in enumerate(all_cts):
        file_idx = file.find("ct")
        sl = iio.imread(file)
        sl_np = np.asarray(sl[:, :, 0])
        # Get index between ct and .png
        name = "ct" + str(file[file_idx + 2:-4])
        name_list.append(name)
        sl_np_list.append(sl_np)
    sl_dict = dict(zip(name_list, sl_np_list))
    file = CT_DIR + "/" + str(filename) + ".npz"
    np.savez_compressed(file, **sl_dict)
    print("Saved successfully")


def save_CT_to_npz_V2(CT_DIR, filename):
    patients = glob.glob(CT_DIR + '/*/')
    patients = sorted(patients, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    sl_np_list = []
    name_list = []
    print("Saving all CT npy arrays to NPZ")
    for idx, patient in enumerate(patients):
        filenames = glob.glob(os.path.join(patient, '*.npy'))
        filenames = list(sorted(filenames))
        for i, file in enumerate(filenames):
            file_idx = file.find("ct")
            sl = np.load(file)
            # Get index between ct and .npy
            name = "ct" + str(file[file_idx + 2:-4])
            name_list.append(name)
            sl_np_list.append(sl)
    sl_dict = dict(zip(name_list, sl_np_list))
    file = CT_DIR + "/" + str(filename) + ".npz"
    np.savez_compressed(file, **sl_dict)
    print("Saved successfully")


def save_masks_to_npz(MASK_DIR, filename):
    all_masks = glob.glob(MASK_DIR + "/*.png")
    print("Saving all Mask arrays to NPZ")
    mask_np_list = []
    name_list = []
    for idx, file in enumerate(all_masks):
        file_idx = file.find("mask")
        mask = iio.imread(file)
        mask_np = np.asarray(mask[:, :, 0])
        # Get index between mask and .png
        name = "mask" + str(file[file_idx + 4:-4])
        name_list.append(name)
        mask_np_list.append(mask_np)
    sl_dict = dict(zip(name_list, mask_np_list))
    file = MASK_DIR + "/" + str(filename) + ".npz"
    np.savez_compressed(file, **sl_dict)
    print("Saved successfully")


def split_folders(CT_DIR, MASK_DIR):
    """
        Split folders such that patient folders are divided into Training, Validation and Test sets
        The function saves 3 separate npz files for the mentioned folders both for CTs and Masks
        The npz files contain a dictionary consisting of cts and masks with the value being the numpy array
        that represents either the CT or the Mask and the Key being the name of the given datapoint
        (i.e mask0_0 which indicates this is the mask of the first slice of the first patient)

        Input: The input directories must have the following structure:
        CTdata OR MaskData
        - CTfolder0 OR Maskfolder0
            - ct0_0.npy OR mask0_0.png
            - ct0_1.npy
            ...
        - CTfolder1
        ...

        The naming scheme of the individual files must be strictly followed.
    """
    # Read all the patient files from the folders
    patientCTS = glob.glob(CT_DIR + '/*/')
    patientCTS = sorted(patientCTS, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    patientMasks = glob.glob(MASK_DIR + '/*/')
    patientMasks = sorted(patientMasks, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    # Zip the cts and masks to maintain correlation
    temp = list(zip(patientCTS, patientMasks))
    # Enumerate to keep index correlation with data
    temp2 = list(enumerate(temp))
    # Set random seed to be reproducible
    random.seed(42)
    # Shuffle the data
    random.shuffle(temp2)
    indices, res = zip(*temp2)
    res1, res2 = zip(*res)
    # res1 and res2 returned as tuples, and so must be converted to lists.
    indices, res1, res2 = list(indices), list(res1), list(res2)
    print(f"CTS after shuffle :  {res1}")
    print(f"Masks after shuffle :  {res2}")
    print(f"Indices after shuffle : {indices}")
    # Define split ratios
    train_perc, val_perc, test_perc = 0.8, 0.1, 0.1
    split_train_idx = int(train_perc * len(indices))
    split_val_idx = split_train_idx + int(math.ceil(val_perc * len(indices)))
    train_patients = indices[:split_train_idx]
    val_patients = indices[split_train_idx:split_val_idx]
    test_patients = indices[split_val_idx:]
    # For next time, I have to loop through the indices in the sets, access the folders based on index, open npy or img
    # and save to different npz files similar to save_CT_to_npz_V2
    # Loop through all the training patients folders
    print("Saving all CT npy arrays and masks to NPZ")
    # Loop through the 3 sets
    for i, patients in enumerate([train_patients, val_patients, test_patients]):
        ct_np_list = []
        ct_name_list = []
        mask_np_list = []
        mask_name_list = []
        for idx in patients:
            # Get the filenames for all the CT npy arrays and masks for a patient
            ct_filenames, mask_filenames = read_patient_cts_masks(res1[idx], res2[idx])
            # Read all the CT npy arrays and masks of that patient
            ctsnames, ctsnp, masksnames, masksnp = prepare_to_save_npz(ct_filenames, mask_filenames)
            # Append this data into lists that will be used to save to npz
            ct_name_list.extend(ctsnames)
            ct_np_list.extend(ctsnp)
            mask_name_list.extend(masksnames)
            mask_np_list.extend(masksnp)

        # Save cts to npz file
        filename = 'ctTrain' if i == 0 else 'ctVal' if i == 1 else 'ctTest'
        sl_dict = dict(zip(ct_name_list, ct_np_list))
        file = CT_DIR + "/" + str(filename) + ".npz"
        np.savez_compressed(file, **sl_dict)
        # Save masks to npz file
        filename = 'maskTrain' if i == 0 else 'maskVal' if i == 1 else 'maskTest'
        sl_dict = dict(zip(mask_name_list, mask_np_list))
        file = MASK_DIR + "/" + str(filename) + ".npz"
        np.savez_compressed(file, **sl_dict)

    print("Saved successfully")


def read_patient_cts_masks(ct, mask):
    """
        Get the filenames for all the CT npy arrays and masks for a patient

        The inputs must be given in a var[idx] format where var is ct or mask and idx is the patient index

        Returns the sorted lists for cts and masks of given patient
    """
    ct_filenames = glob.glob(os.path.join(ct, '*.npy'))
    ct_filenames = list(sorted(ct_filenames))
    mask_filenames = glob.glob(os.path.join(mask, '*.png'))
    mask_filenames = list(sorted(mask_filenames))
    return ct_filenames, mask_filenames


def prepare_to_save_npz(ct_filenames, mask_filenames):
    """
        Read all the CT npy arrays and masks for a patient

        Input: CT filenames and Mask filenames for that given patient

        Return: A list that contains the names and a list that contains the npy data for both cts and masks
    """
    ct_np_list = []
    ct_name_list = []
    mask_np_list = []
    mask_name_list = []
    for i, file in enumerate(ct_filenames):
        file_idx = file.find("ct")
        sl = np.load(file)
        # Get index between ct and .npy
        name = "ct" + str(file[file_idx + 2:-4])
        ct_name_list.append(name)
        ct_np_list.append(sl)

    for idx, file in enumerate(mask_filenames):
        file_idx = file.find("mask")
        mask = iio.imread(file)
        mask_np = np.asarray(mask[:, :, 0])
        # Get index between mask and .png
        name = "mask" + str(file[file_idx + 4:-4])
        mask_name_list.append(name)
        mask_np_list.append(mask_np)

    return ct_name_list, ct_np_list, mask_name_list, mask_np_list
    # train_patientsCT = res1[:split_train_idx]
    # val_patientsCT = res1[split_train_idx:split_val_idx]
    # test_patientsCT = res1[split_val_idx:]
    # train_patientsMasks = res2[:split_train_idx]
    # val_patientsMasks = res2[split_train_idx:split_val_idx]
    # test_patientsMasks = res1[split_val_idx:]

# def remove_underscore(CT_DIR,MASK_DIR):
#    all_masks = glob.glob(MASK_DIR+"/*.png")
#    all_cts = glob.glob(CT_DIR+"/*.png")
#    for idx, old_file in enumerate(all_masks):
#        name = old_file.replace("_","")
#        new_file = str(name)
#        print(new_file)
#        os.rename(old_file,new_file)   
#        
#    for idx, old_file in enumerate(all_cts):
#        name = old_file.replace("_","")
#        new_file = str(name)
#        #print(new_file)
#        os.rename(old_file,new_file)


# %% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    MASK_DIR = '/home/ERASMUSMC/099035/Documents/MasksV2'
    CT_DIR = '/home/ERASMUSMC/099035/Documents/CTimagesV2'

    # save_CT_to_npz(CT_DIR,filename = "ct_npz")
    # save_masks_to_npz(MASK_DIR,filename = "mask_npz")
    # save_CT_to_npz_V2(CT_DIR, filename='ctv2_npz')

    split_folders(CT_DIR,MASK_DIR)

    # # Load the saved npz
    # npz_cts = np.load(CT_DIR + "/ct_npz.npz")
    # npz_masks = np.load(MASK_DIR + "/mask_npz.npz")
    # # Compare the npz files with the originals
    # # keep in mind all_cts are randomized because of reading them from glob
    # insp = npz_cts['ct0_10']
    # fig, axs = plt.subplots(1, 2)
    #
    # axs[0].imshow(insp, cmap='gray')
    # axs[0].set_title("ct0_10")
    # insp_mask = npz_masks['mask0_23']
    # axs[1].set_title("mask0_23")
    # axs[1].imshow(insp_mask, cmap='gray')
    # # remove_underscore(CT_DIR,MASK_DIR) #DD NOT USE

    # all_cts = glob.glob(CT_DIR+"/*.png")


if __name__ == "__main__":
    main()
