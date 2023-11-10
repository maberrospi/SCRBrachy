#!/usr/bin/env python
# coding: utf-8

# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import ipywidgets as widgets-Cannot be used in spyder
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'ipympl')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider
import imageio.v2 as imageio
from pydicom import dcmread
from tqdm import tqdm
import timeit
from ImageData import ImageData
# import imageio.v3 as iio
import os, glob


# %% DEFINE ALL FUNCTIONS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def list_dicoms(dicom_dir, filename_expression='*.dcm'):
    filenames = glob.glob(os.path.join(dicom_dir, filename_expression))
    filenames = sorted(filenames)
    print(f'\nThere are {len(filenames)} DICOMs (slices) in the directory "{dicom_dir}"')
    return list(filenames)


def show_single_slice(filename):
    # Read dicom file
    sl = imageio.imread(filename)
    Coronal_pixel_space, Saggital_pixel_space = sl.meta['PixelSpacing']
    print(f'Pixel spacing values:\n\tCoronal = {Coronal_pixel_space}mm\n\tSaggital = {Saggital_pixel_space}mm')
    sl_np = np.asarray(sl)
    print(type(sl_np))
    # Show the image with a gray colormap
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(sl_np, cmap='gray')
    ax.axis('off')
    ax.set_title('Axial slice')
    plt.show()


def show_single_slice_with_mask(filename, x, y):
    # Read dicom file
    sl = imageio.imread(filename)
    Coronal_pixel_space, Saggital_pixel_space = sl.meta['PixelSpacing']
    print(f'Pixel spacing values:\n\tCoronal = {Coronal_pixel_space}mm\n\tSaggital = {Saggital_pixel_space}mm')
    # Create an RGB array from the given image
    # sl_RGB = np.dstack([sl,sl,sl])
    # Show the annotations with red color if the x,y data exists  
    # sl[y,x] = 255
    # Show the image with a gray colormap
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(sl, cmap='gray')  # ,norm='linear'
    if len(x) != 0 and len(y) != 0:
        # sl_RGB[y,x] = [255,0,0]
        plt.scatter(x, y, color="red", marker='s', s=72. / fig.dpi)
    ax.axis('off')
    ax.set_title('Axial slice')
    plt.show()


# Define the function that shows the images of the specified slice number.
# It starts with the 50th slice. And you can scroll over any slice
# using the slider.
def axial_slicer(allslices, axial_slice=1):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(bottom=0.25)
    # Show the image of the specified slice number in 'gray' color-map
    # and axial aspect ratio
    # Add a slider that starts with 0 and ends at the number of slices
    axslider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(axslider, "Slice", valmin=0, valmax=len(allslices) - 1, valinit=1)
    ax.imshow(allslices[slider.val - 1], cmap='gray')
    # Don't show the axis
    ax.axis('off')
    slider.on_changed(update)
    plt.show()


def update(val):
    axial_slicer(val)


def show_all_slices(filenames):
    # Read all slices
    allslices = []
    for i in range(0, len(filenames)):
        allslices.append(imageio.imread(filenames[i]))
    axial_slicer(allslices)
    # Add a slider that starts with 0 and ends at the number of slices
    # widgets.interact(axial_slicer,axial_slice=(0,filenames_size-1))


def find_slice_locations(filenames):
    Zslices = []
    for file in filenames:
        sl = dcmread(file)
        Zslices.append(sl.SliceLocation)
    shape = sl.pixel_array.shape
    return Zslices, shape


def sort_CT_folder(folder):
    filenames_slices = list_dicoms(folder)
    sl_list = []
    for file in filenames_slices:
        sl = imageio.imread(file)
        sl_list.append(sl)
        # print(str(sl.meta['ImagePositionPatient'][2])+"\n")
        # print(file)
    # Sort the list based on instance number
    dct = dict(zip(filenames_slices, sl_list))
    sorted_dct = dict(sorted(dct.items(), key=lambda item: item[1].meta['ImagePositionPatient'][2]))
    # Save the new sorted files by renaming the old ones
    # file is the key of the dict
    for idx, old_file in enumerate(sorted_dct):
        name = "CT_slice" + str(idx)
        new_file = folder + "/" + str(name) + ".dcm"
        os.rename(old_file, new_file)
        # Check if correct


#    for key, value in sorted_dct.items() :
#        print (key, value.meta['ImagePositionPatient'][2])

def save_CT_folder(CT_DIR, folder, folder_number):
    """Saves CT slices from ONE given folder"""
    filenames_slices = list_dicoms(folder)
    filenames_slices = sorted(filenames_slices, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    for index, file in enumerate(filenames_slices):
        sl = imageio.imread(file)
        name = "ct" + str(folder_number) + "_" + str(index)
        file = CT_DIR + "/" + str(name) + ".png"
        plt.imsave(file, sl, cmap='gray')


def save_CT_slices(CT_DIR, folders):
    """Saves ALL the CT slices from a given list of folders"""
    for idx, folder in enumerate(tqdm(folders, desc="CT folder")):
        filenames_slices = list_dicoms(folders[idx])
        for index, file in enumerate(filenames_slices):
            sl = imageio.imread(file)
            name = "ct" + str(idx) + "_" + str(index)
            file = CT_DIR + "/" + str(name) + ".png"
            plt.imsave(file, sl, cmap='gray')


def find_min_max(folders):
    """Find the min and max values from all the CT slices"""
    slice_list = []
    for idx, folder in enumerate(tqdm(folders, desc="CT folder")):
        filenames_slices = list_dicoms(folder)
        for index, file in enumerate(filenames_slices):
            sl = imageio.imread(file)
            slice_list.append(sl)
    if slice_list:
        hmin = np.min(slice_list)
        hmax = np.max(slice_list)
        print(f'images min: {hmin}\nimages max: {hmax}')
        return hmin, hmax
    else:
        print("The list provided is empty")
        return None, None


def save_CT_slices_v2(CT_DIR, folders, hmin=-1024, hmax=3071):
    """Saves ALL the CT slices from a given list of folders into patient folders in npy format"""
    for idx, folder in enumerate(tqdm(folders, desc="CT folder")):
        filenames_slices = list_dicoms(folder)
        # Check if folder to save does not exist
        if not os.path.exists(CT_DIR):
            os.mkdir(CT_DIR)
        if not os.path.exists(os.path.join(CT_DIR, os.path.basename(os.path.normpath(folder)))):
            os.mkdir(os.path.join(CT_DIR, os.path.basename(os.path.normpath(folder))))
        for index, file in enumerate(filenames_slices):
            # Imageio pixel array is in hounsfield units
            # https://towardsdatascience.com/dealing-with-dicom-using-imageio-python-package-117f1212ab82
            sl = imageio.imread(file)
            # Perform min max normalization
            # Multiply by 255 just to get the values between 0 and 255
            #sl = (sl - hmin) * 255 / (hmax - hmin)
            #sl = sl.astype(np.uint8)
            name = "ct" + str(idx) + "_" + str(index)
            file = CT_DIR + "/" + os.path.basename(os.path.normpath(folder)) + "/" + str(name) + ".npy"
            # Save to npy files
            np.save(file, sl)
            #im = Image.fromarray(sl)
            # Convert to grayscale
            #im = im.convert('L')
            #im.save(file)
            # plt.imsave(file, sl, cmap='gray')


# %% MAIN CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    # Read all DICOM files (CT Slices)
    DICOM_DIR = '/home/ERASMUSMC/099035/Documents/DICOMfiles'
    CT_DIR = '/home/ERASMUSMC/099035/Documents/CTimagesV2'
    folders = glob.glob(DICOM_DIR + '/*')
    folders = sorted(folders, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    filenames = list_dicoms(folders[15])
    filenames = sorted(filenames, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    filenames_size = len(filenames)

    # Turns out imageio is faster when reading the DICOMS, but also has less info.
    # print(timeit.timeit('imageio.imread(filename)',setup ='import imageio.v2 as imageio; filename = "/home/ERASMUSMC/099035/Documents/DICOMfiles/CT0/1.3.12.2.1107.5.1.4.100037.30000019022608594181000005109.dcm"' ,number=1000))
    # print(timeit.timeit('dcmread(filename)',setup = 'from pydicom import dcmread; filename = "/home/ERASMUSMC/099035/Documents/DICOMfiles/CT0/1.3.12.2.1107.5.1.4.100037.30000019022608594181000005109.dcm"',number=1000))
    # Show a single slice
    # show_single_slice(filenames[34])
    # save_CT_folder(CT_DIR,folders[15],15)
    # save_CT_slices(CT_DIR,folders)
    # Renames ALL files in the given folder
    # Might be good to add this in both save CT slices and CT folder
    # Uncomment ONLY if you are sure you want to do this
    # sort_CT_folder(folders[15])

    # Find min max of all CT slices
    # hmin, hmax = find_min_max(folders)
    # Testing saving v2
    save_CT_slices_v2(CT_DIR, folders)


if __name__ == "__main__":
    main()
# ApplicationSetupSequence[0].ChannelSequence[0].BrachyControlPointSequence[41].ControlPoint3DPosition
# slice.meta
# slice.meta['PixelData']
# The 2 dimensional array contains numbers where each
# number represents the pixel value in Hounsfield Unit(HU)
# slice.meta['shape']

# Show a single slice

# show_single_slice(filenames[15])

# Show all slices (need Spyder 4.0 for interactivity)
# show_all_slices(filenames)

# %% Testing to read CT slices using pydicom

# Zslices = find_slice_locations(filenames)
# print(Zslices)
#
## Print single slice
# slicet = dcmread(filenames[50])
# plt.imshow(slicet.pixel_array, cmap = 'gray')
# plt.axis('off')
# plt.title('Axial slice')
# plt.show()

# %% Testing Image Data
# ImgData = ImageData(filenames[0])
# ImgData.VisAxial(25)
