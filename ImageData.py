# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import pydicom
import matplotlib.pyplot as plt


# %% Define all functions and classes here %%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ImageData:
    def __init__(self, file):
        collection = pydicom.dcmread(file)
        # Read in the DICOM volume
        # print(collection.ImagePositionPatient[0])
        self.volume = np.squeeze(collection.pixel_array).astype(np.float32)
        self.xaxis = np.arange(
            collection.ImagePositionPatient[0],
            collection.PixelSpacing[0] * (self.volume.shape[1])
            + collection.ImagePositionPatient[0],
            collection.PixelSpacing[0],
        )
        self.yaxis = np.arange(
            collection.ImagePositionPatient[1],
            collection.PixelSpacing[1] * (self.volume.shape[0])
            + collection.ImagePositionPatient[1],
            collection.PixelSpacing[1],
        )
        self.zaxis = collection.ImagePositionPatient[2]

    def VisAxial(self, slicenumber):
        # Visualize Registration
        plt.imshow(self.volume[:, :], cmap="gray", aspect="auto")
        plt.show()


# %% MAIN CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    img = ImageData("example_patient_dicom.dcm")


if __name__ == "__main__":
    main()
