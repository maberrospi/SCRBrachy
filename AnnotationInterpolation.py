#!/usr/bin/env python
# coding: utf-8

# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# get_ipython().run_line_magic('matplotlib', 'ipympl')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pydicom import dcmread
import os, glob
from tqdm import tqdm
from Read_Visualize_DICOMS import list_dicoms, find_slice_locations, show_single_slice_with_mask
from ImageData import ImageData

from scipy.interpolate import interp1d


# %% DEFINE ALL FUNCTIONS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def list_annotations(annotation_dir, filename_expression='*.dcm'):
    # does something with annotations
    filenames = glob.glob(os.path.join(annotation_dir, filename_expression))
    filenames = sorted(filenames, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    print(f'There are {len(filenames)} annotations in the directory "{annotation_dir}"')
    return list(filenames)


# Create a class for the Dwell Data
class DwellData:
    channel = np.array([np.nan])
    position = np.array([np.nan])
    coordinates = np.array([0])

    def __init__(self, filename):
        annotations_metadata = dcmread(filename)
        # ApplicationSetupSequence is of Class type 'Sequence' from pydicom docs
        channel_seq = annotations_metadata.ApplicationSetupSequence[0].ChannelSequence
        k = 0
        for c in range(0, len(channel_seq)):
            control_point_seq = channel_seq[c].BrachyControlPointSequence
            relative_position = float('NaN')
            # Initialize arrays if its the first pass
            # Checks if any of the values evaluates to True (non zero)
            is_nan = np.isnan(self.channel)
            if (np.all(is_nan) == True):
                # Initialize numpy arrays with zeros (1D,1D,3D)
                self.channel = np.full((len(channel_seq) * len(control_point_seq)), np.nan)
                self.position = np.full((len(channel_seq) * len(control_point_seq)), np.nan)
                self.coordinates = np.zeros([len(channel_seq) * len(control_point_seq), 3])
            # Loop in the BrachyControlPointSequence from index len(seq)-1 to 0
            for i in range(len(control_point_seq) - 1, -1, -1):
                if (control_point_seq[i].ControlPointRelativePosition != relative_position):
                    relative_position = control_point_seq[i].ControlPointRelativePosition
                    self.channel[k] = c + 1
                    self.position[k] = relative_position
                    self.coordinates[k, :] = control_point_seq[i].ControlPoint3DPosition
                    k += 1
        self.channel = self.channel[np.isnan(self.channel) == False].astype('int32')
        self.position = self.position[np.isnan(self.position) == False].astype('int32')
        self.coordinates = self.coordinates[self.coordinates != 0].reshape((-1, 3))


def DwellOnSliceInterpolation(PointData, Zslices):
    # Define empty cloud
    TotalCloud = []

    # iterate through all catheters
    for i in np.unique(PointData.channel):
        xyz = PointData.coordinates[np.where(PointData.channel == i)]

        # Fit spline through point cloud
        if len(xyz) > 1:
            # Find Zslices within the range of z for the current channel
            Zslices_in_range = [z for z in Zslices if min(xyz[:, 2]) <= z <= max(xyz[:, 2])]

            if Zslices_in_range:  # Check if there are any Zslices in range
                ppx = interp1d(xyz[:, 2], xyz[:, 0], kind='cubic')
                ppy = interp1d(xyz[:, 2], xyz[:, 1], kind='cubic')
                # print(Zslices_in_range)
                # Evaluate spline on new grid
                xyznew = np.zeros((len(Zslices_in_range), 3))
                xyznew[:, 0] = ppx(Zslices_in_range)
                xyznew[:, 1] = ppy(Zslices_in_range)
                xyznew[:, 2] = Zslices_in_range

                # Put data in cloud
                TotalCloud.append(xyznew)

        else:
            # For a single point, check if its z value is in Zslices
            if xyz[0, 2] in Zslices:
                xyznew = np.tile(xyz, (len(Zslices), 1))
                TotalCloud.append(xyznew)

    # turn into single numpy array
    TotalCloud = np.vstack(TotalCloud)

    return TotalCloud


def visualize_interpolated_annotations(PointData, TotalCloud):
    # and visualize our accomplishments!
    # Plot both clouds in 3D
    fig = plt.figure(1, figsize=(6, 6))
    ax = plt.axes(projection="3d")
    ax.scatter3D(PointData.coordinates[:, 0], PointData.coordinates[:, 1], PointData.coordinates[:, 2], s=10)
    ax.scatter3D(TotalCloud[:, 0], TotalCloud[:, 1], TotalCloud[:, 2], color="red", s=30, marker='x')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Annotations")
    plt.show()


def plot_slice_points(slice_points, sl_number):
    # Plot points of #ct slice
    slice_points = slice_points
    fig = plt.figure(2, figsize=(6, 6))
    plt.scatter(slice_points[:, 0], slice_points[:, 1], color="red", marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Z = {sl_number} mm')
    plt.show()


def create_and_save_mask(slice_points, ct_slice, shape, MASK_DIR, name='Test', patient_num=0, save=True):
    """
    Function that creates and saved the mask in its appropriate folder

    Some changes have been made for V2 named here:
        1. Added new parameter to pass the patient number called patient_num to save in appropriate folder
        2. Changed the 'file' variable to add this patient number and set folder name

    """
    # X and Y interpolated positions of the specific slice at position Z
    Xpositions = slice_points[:, 0]
    Ypositions = slice_points[:, 1]
    # Load the x and y axis range from the original CT slice
    img = ImageData(ct_slice)
    # Find all the X pixel positions
    x_pixels_list = []
    for x in Xpositions:
        x_pixels_list.append(np.where(np.power(img.xaxis - x, 2) == np.min(np.power(img.xaxis - x, 2))))
    x_pixels = np.squeeze(x_pixels_list)

    # Find all the Y pixel positions
    y_pixels_list = []
    for y in Ypositions:
        y_pixels_list.append(np.where(np.power(img.yaxis - y, 2) == np.min(np.power(img.yaxis - y, 2))))
    y_pixels = np.squeeze(y_pixels_list)

    # Check if pixel arrays are not empty
    if y_pixels.size > 0 and x_pixels.size > 0:
        # Expand x and y pixels to 3x3 box of pixels
        # Every 9 values in x and y arrays represent a 3x3 box in the image
        x_pixels3x3 = x_pixels
        y_pixels3x3 = y_pixels
        # Check if x and y pixels are iterable
        try:
            iter(x_pixels)
        except TypeError as te:
            # If exception is thrown calculate only the one value
            x = x_pixels
            y = y_pixels
            pos = 0
            x_exp = np.array([x + 1, x - 1, x, x, x + 1, x + 1, x - 1, x - 1])
            x_pixels3x3 = np.insert(x_pixels3x3, pos, x_exp)
            y_exp = np.array([y, y, y + 1, y - 1, y + 1, y - 1, y + 1, y - 1])
            y_pixels3x3 = np.insert(y_pixels3x3, pos, y_exp)
        else:
            for idx, x in enumerate(x_pixels):
                pos = idx + idx * 8
                x_exp = np.array([x + 1, x - 1, x, x, x + 1, x + 1, x - 1, x - 1])
                x_pixels3x3 = np.insert(x_pixels3x3, pos, x_exp)
            for idx, y in enumerate(y_pixels):
                pos = idx + idx * 8
                y_exp = np.array([y, y, y + 1, y - 1, y + 1, y - 1, y + 1, y - 1])
                y_pixels3x3 = np.insert(y_pixels3x3, pos, y_exp)

    # Create numpy array with zeros of image size
    mask = np.zeros(shape)
    # If the x,y pixels are not empty, create mask and save
    # If they are empty leave a totally black mask and save it
    if y_pixels.size > 0 and x_pixels.size > 0:
        mask[y_pixels3x3, x_pixels3x3] = 255
    else:
        x_pixels3x3 = []
        y_pixels3x3 = []
    if save:
        if not os.path.exists(MASK_DIR):
            os.mkdir(MASK_DIR)
        if not os.path.exists(os.path.join(MASK_DIR, "MASKS" + str(patient_num))):
            os.mkdir(os.path.join(MASK_DIR, "MASKS" + str(patient_num)))
        file = MASK_DIR + "/" + "MASKS" + str(patient_num) + "/" + str(name) + ".png"
        plt.imsave(file, mask, cmap='gray')

    return mask, x_pixels3x3, y_pixels3x3


def create_and_save_folder_masks(annotation, ct_folder, ct_folder_number, MASK_DIR):
    """
    Create and save masks from ONE folder

    Some changes have been made for V2 named here:
        1. Used the ct_folder_number parameter to save the mask in the respective patient folder
        2. Pass this parameter in create_and_save_mask to pass this number called patient_num

    """

    print('Creating all masks for the CT images in the folder given using the Annotated interpolation data')
    # Create an object of class DwellData
    PointData = DwellData(annotation)
    # Read all CT slice dicoms for given annotation file
    filenames_slices = list_dicoms(ct_folder)
    filenames_slices = sorted(filenames_slices, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    # Extract the Z slice locations and the shape of the image
    Zslices, shape = find_slice_locations(filenames_slices)
    # Interpolate Dwell Points on slices
    TotalCloud = DwellOnSliceInterpolation(PointData, Zslices)
    for index, slice_number in enumerate(tqdm(Zslices, desc="Mask Number")):
        slice_points = TotalCloud[TotalCloud[:, 2] == slice_number]
        name = "mask" + str(ct_folder_number) + "_" + str(index)
        # Create the mask, save it, and return its values and the x,y pixel positions (x,y pixel are only for visualization on original CT)
        mask, x, y = create_and_save_mask(slice_points, filenames_slices[index], shape, MASK_DIR, name=name, patient_num=ct_folder_number)


def create_and_save_all_folder_masks(annotation_files, ct_folders, MASK_DIR):
    """
    Create and save ALL masks

    Some changes have been made for V2 named here:
        1. Added folder_number to use for saving masks in respective patient folder
        2. Added new parameter in create_and_save_mask to pass this number called patient_num

    """
    # If the size of annotations and CT folders doesnt match then there is an error
    filenames_size = len(annotation_files)
    folders_size = len(ct_folders)
    if filenames_size == folders_size:
        # Folder number to use when saving all the annotations in their respective patient folder
        folder_number = 0
        print("Creating all the masks for the CT images using the Annotated interpolation data")
        for idx, annotation in enumerate(tqdm(annotation_files, desc="Creating Masks")):
            # Create an object of class DwellData
            PointData = DwellData(annotation)
            # Read all CT slice dicoms for given annotation file
            filenames_slices = list_dicoms(ct_folders[idx])
            filenames_slices = sorted(filenames_slices, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
            # Extract the Z slice locations and the shape of the image
            Zslices, shape = find_slice_locations(filenames_slices)
            # Interpolate Dwell Points on slices
            TotalCloud = DwellOnSliceInterpolation(PointData, Zslices)
            # Create a masks for all slices
            for index, slice_number in enumerate(tqdm(Zslices, desc="Mask Number")):
                slice_points = TotalCloud[TotalCloud[:, 2] == slice_number]
                name = "mask" + str(idx) + "_" + str(index)
                # Create the mask, save it, and return its values and the x,y pixel positions (x,y pixel are only for visualization on original CT)
                mask, x, y = create_and_save_mask(slice_points, filenames_slices[index], shape, MASK_DIR, name=name, patient_num=folder_number)

            folder_number += 1
    else:
        print("The annotation and CT folder sizes do NOT match")


def plot_mask(mask):
    fig = plt.figure(3, figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.show()


# %% MAIN CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():
    # some dummy data
    VisualizeSliceNumber = 40  # which channel to plot in 2D
    # Read DICOM annotations
    ANNOTATION_DIR = "/home/ERASMUSMC/099035/Documents/AnnotationFiles"
    DICOM_DIR = '/home/ERASMUSMC/099035/Documents/DICOMfiles'
    MASK_DIR = '/home/ERASMUSMC/099035/Documents/MasksV2'
    # Read filenames and find total number for files
    filenames = list_annotations(ANNOTATION_DIR)
    folders = glob.glob(DICOM_DIR + '/*')
    folders = sorted(folders, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))

    # # Create an object of class DwellData
    # PointData = DwellData(filenames[9])
    #
    # # Read Z location from CT slices
    # folders = glob.glob(DICOM_DIR + '/*')
    # folders = sorted(folders, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    # filenames_slices = list_dicoms(folders[9])
    # filenames_slices = sorted(filenames_slices, key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
    # Zslices, shape = find_slice_locations(filenames_slices)
    #
    # # Example code of how Interpolation and Mask creation is performed
    # # Interpolate Dwell Points on slices
    # TotalCloud = DwellOnSliceInterpolation(PointData, Zslices)
    # # Visualize the interpolated data
    # visualize_interpolated_annotations(PointData, TotalCloud)
    # # Plot the x,y points for a specific slice
    # slice_points = TotalCloud[TotalCloud[:, 2] == Zslices[VisualizeSliceNumber]]
    # plot_slice_points(slice_points, Zslices[VisualizeSliceNumber])
    # # Create the mask, save it, and return its values and the x,y pixel positions (x,y pixel are only for visualization on original CT)
    # mask, x, y = create_and_save_mask(slice_points, filenames_slices[VisualizeSliceNumber], shape, MASK_DIR,
    #                                   name="TestMask", save=False)
    # # Plot the calculated mask
    # plot_mask(mask)
    # # Show the mask on the original CT slice
    # show_single_slice_with_mask(filenames_slices[VisualizeSliceNumber], x, y)

    # Create and save masks from one CT folder
    # create_and_save_folder_masks(filenames[15],folders[15],15,MASK_DIR)

    # Create and save ALL masks
    create_and_save_all_folder_masks(filenames, folders, MASK_DIR)


if __name__ == "__main__":
    main()
