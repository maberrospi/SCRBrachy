# %% Import libraries HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os, glob
from shutil import copytree, copy2


# %% DEFINE ALL FUNCTIONS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_CT_folder(CTfolders, DICOM_DIR):
    i = 0
    print(f"Creating all CT folders in {DICOM_DIR}")
    for lst in CTfolders:
        # Convert single item of the list into a string
        lst = "".join(lst)
        path = DICOM_DIR + "/CT" + str(i)
        # If the directory already exists the command does NOT overwrite existing files -> distutils.copy_tree
        # copy_tree(lst,path)
        # If the directory already exists files within the tree will be overwritten
        copytree(lst, path, dirs_exist_ok=True)
        i += 1
    print("All CT folders and their data succesfully coppied")


def create_Annotation_folder(Annotationfiles, ANNOTATION_DIR):
    i = 0
    print(f"Copying all Annotation data in {ANNOTATION_DIR}")
    for lst in Annotationfiles:
        lst = "".join(lst)
        path = ANNOTATION_DIR + "/ANN" + str(i) + ".dcm"
        # If the file already exists it will be overwritten
        copy2(lst, path)
        i += 1
    print("All files successfully copied")


# %% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    ANNOTATION_DIR = "/home/ERASMUSMC/099035/Documents/AnnotationFiles"
    DICOM_DIR = "/home/ERASMUSMC/099035/Documents/DICOMfiles"
    ALLDATA_DIR = "/home/ERASMUSMC/099035/Documents/ForAItraining/prostate"
    # Read all the folders in the prostate folder and sort them
    CTfolders = []
    Annotationfiles = []
    folders = glob.glob(ALLDATA_DIR + "/*")
    folders = sorted(folders)
    # Read all the subfolders in the prostate folder and divide into CT folders
    # and EMT data (Annotations)
    for folder in folders:
        if len(os.listdir(folder)) != 0:
            subfolders = glob.glob(folder + "/*")
            for subfolder in subfolders:
                if len(os.listdir(subfolder)) != 0:
                    CTfolders.append(glob.glob(subfolder + "/*[!.dcm]"))
                    Annotationfiles.append(
                        glob.glob(os.path.join(subfolder, "*EMT*.dcm"))
                    )

    # Remove all unecessary files from the CT folders
    for ctidx, CTlist in enumerate(CTfolders):
        to_remove = []
        for idx, item in enumerate(CTlist):
            if not os.path.isdir(item):
                to_remove.append(CTfolders[ctidx][idx])
        for item in to_remove:
            # print(item)
            CTfolders[ctidx].remove(item)

    # Create new directories and add all the CT data and Annotations
    # with appropriate names to map them to each other

    create_CT_folder(CTfolders, DICOM_DIR)
    create_Annotation_folder(Annotationfiles, ANNOTATION_DIR)


if __name__ == "__main__":
    main()
