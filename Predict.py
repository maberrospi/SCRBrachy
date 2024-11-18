# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob
import re
from PIL import Image
import timeit
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from Metrics import multiclass_dice_coeff, dice_coeff, dice_loss
from Train import (
    CTCatheterDataset,
    train_cts,
    train_masks,
    val_cts,
    val_masks,
    NormalizationMinMax,
    ToTensor,
    load_data_npz,
)
from UNet import UNet
from AttUNet import AttUNet


def predict_dice_score(
    model, dataset, device, threshold=0.5, train_img_size=(512, 512)
):
    """
    @param model: Model with loaded parameters
    @param dataset: DataLoader instance of the set you want to calculated the DSC
    @param device: CPU or GPU
    @param threshold: Given threshold for predicting positive and negative pixels (models trained with 0.5)
    @param train_img_size: Image size used for down/up sampling
    @return: Mean DSC of the given dataset
    """
    model.eval()
    dice_score = 0
    dice_list = []
    for batch in tqdm(dataset, desc="Image"):
        image, true_mask = batch["image"], batch["mask"]
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(image).cpu()
            # Down/up sample the input image to the required size if it's not
            output = F.interpolate(output, train_img_size, mode="bilinear")
            mask = torch.sigmoid(output) > threshold
            batch_dice = dice_coeff(mask, true_mask, reduce_batch_first=False)
            dice_score += batch_dice
            dice_list.append(batch_dice)
    dice_score = dice_score / len(dataset)
    dice_score_std = torch.std(torch.as_tensor(dice_list))
    print(f"Final dice score: {dice_score}")
    print(f"Standard deviation: {dice_score_std}")
    return dice_score


def predict_image(model, img, device, orig, threshold=0.5, train_img_size=(512, 512)):
    """
    @param model: Model with loaded parameters
    @param img: Numpy array containing HU values for a given CT slice
    @param device: CPU or GPU
    @param orig: Ground truth mask image loaded using PIL
    @param threshold: Given threshold for predicting positive and negative pixels (models trained with 0.5)
    @param train_img_size: Image size used for down/up sampling
    @return: Predicted mask as a numpy array

    """
    model.eval()
    # Pre-process the given ct slice
    image = CTCatheterDataset.preprocess(img)
    # Unsqueeze as the model expects BxCxHxW
    image = torch.unsqueeze(image, 0)
    img = image.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        # Move all model parameters and buffers to the cpu
        output = model(img).cpu()
        orig = np.asarray(orig)
        # Extract single channel from RGBA
        orig = orig[:, :, 0]
        # Normalize grayscale mask
        orig = orig / 255
        orig = np.expand_dims(orig, axis=0)
        orig = torch.from_numpy(orig)
        output1 = torch.sigmoid(output) > threshold
        print(dice_coeff(torch.squeeze(output1, dim=0), orig))
        # Down/up sample the input image to the required size if it's not
        output = F.interpolate(output, train_img_size, mode="bilinear")
        if model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > threshold
            # Squeeze to remove batch dimension and channel
            mask = torch.squeeze(mask, dim=(0, 1))
            # Convert mask to numpy
            mask = mask.numpy()
    return mask


def plot_img_and_mask(image, mask, orig_mask):
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(orig_mask, cmap="gray")
    axs[2].imshow(mask, cmap="gray")
    axs[0].set_title("CT slice")
    axs[1].set_title("Original mask")
    axs[2].set_title("Predicted mask")
    fig.tight_layout()
    plt.show()


def plot_mask_on_img(image, mask, orig_mask):
    # Plot original ct slice
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].imshow(image, cmap="gray")
    # Check if the predicted mask contains any annotations
    if np.any(mask):
        axs[0].contour(mask, alpha=0.7, colors="red", corner_mask=False)
    orig_m = np.asarray(orig_mask)
    # Extract single channel from RGBA
    orig_m = orig_m[:, :, 0]
    # Normalize grayscale mask
    orig_m = orig_m / 255
    axs[1].imshow(mask - orig_m, cmap="gray", vmin=-1, vmax=1)
    axs[0].axis("off")
    axs[1].axis("off")
    axs[0].set_title("Prediction on CT slice")
    axs[1].set_title("Prediction vs Original mask difference")
    fig.tight_layout()
    plt.show()


def prediction_time():
    """
    This function calculated the time needed to perform a single prediction
    The setup string includes the model, model checkpoints and slice/mask paths which should be changed according
        to what time you are looking for
    The time is calculated based on the time needed to run the predict_image() function
    """
    SETUP = """
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import timeit
from Metrics import multiclass_dice_coeff, dice_coeff, dice_loss
from Train import CTCatheterDataset, train_cts, train_masks, val_cts, val_masks, NormalizationMinMax, ToTensor, load_data_npz
from Model import UNet
from AttUNet import AttUNet
from __main__ import predict_image
#model = UNet(n_channels=1, n_classes=1)
model = AttUNet(n_channels=1, n_classes=1)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model = '/home/ERASMUSMC/099035/Desktop/PythonWork/Baseenv/checkpoints/unetHyperOptES10/checkpoint_epoch27.pth'
attunet_model = '/home/ERASMUSMC/099035/Desktop/PythonWork/Baseenv/checkpoints/attunetHyperOptES10/checkpoint_epoch22.pth'
# Load the trained model
# Change between unet and attunet appropriately
model.load_state_dict(
    torch.load(attunet_model, map_location=device))

# # Run inference on a single image
path = "/home/ERASMUSMC/099035/Documents/MasksV2/MASKS14/mask14_24.png"
path2 = "/home/ERASMUSMC/099035/Documents/CTimagesV2/CT14/ct14_24.npy"
assert os.path.exists(path) or os.path.exists(path2), "One of the files does not exist"
mask = Image.open(path)
# img = Image.open(path2)
img = np.load(path2)
        """

    TEST_CODE = """
pred_mask = predict_image(model, img, device, mask)
        """
    times = timeit.repeat(setup=SETUP, stmt=TEST_CODE, repeat=3, number=1000)

    # printing minimum exec. time
    print("All prediction min time: {}".format(min(times)))
    print("Singular prediction time: {}".format(min(times) / 1000))


def find_non_TN(model, pat_dir_ct, pat_dir_mask, device, threshold=0.5):
    """
    Return a list of all the predictions in a single patient that are not True Negative predictions
    This means that we disregard all black mask predictions with black mask ground truths.
    This function serves its purpose when qualitatively looking at the results to determine TP,FP and FN
    """
    model.eval()
    inspect_list = []
    cts_list = list_cts(pat_dir_ct)
    masks_list = list_masks(pat_dir_mask)
    cts_sorted, masks_sorted = sort_alphanumerically(cts_list, masks_list)
    for index, files in enumerate(tqdm(zip(cts_sorted, masks_sorted), desc="Slice")):
        assert os.path.exists(files[0]) or os.path.exists(
            files[1]
        ), "One of the files does not exist"
        mask = Image.open(files[1])
        img = np.load(files[0])
        image = CTCatheterDataset.preprocess(img)
        # Unsqueeze as the model expects BxCxHxW
        image = torch.unsqueeze(image, 0)
        img = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            # Move all model parameters and buffers to the cpu
            output = model(img).cpu()
            orig = np.asarray(mask)
            orig = orig[:, :, 0]
            orig = orig / 255
            output1 = torch.sigmoid(output) > threshold
            # Convert mask to numpy
            output1 = output1.numpy()
        if np.any(orig) or np.any(output1):
            # If any mask contains any annotations save its basename in a list
            inspect_list.append(os.path.splitext(os.path.basename(files[0]))[0])
    print(f"The slices to be further inspected are: {inspect_list}")
    return inspect_list


def list_cts(pat_dir, filename_expression="*.npy"):
    """
    Helper function for find_non_TN that lists cts and masks of a single patient
    """
    filenames = glob.glob(os.path.join(pat_dir, filename_expression))
    filenames = sorted(filenames)
    print(f'\nThere are {len(filenames)} CT slices in the directory "{pat_dir}"')
    return list(filenames)


def list_masks(pat_dir, filename_expression="*.png"):
    """
    Helper function for find_non_TN that lists cts and masks of a single patient
    """
    filenames = glob.glob(os.path.join(pat_dir, filename_expression))
    filenames = sorted(filenames)
    print(f'\nThere are {len(filenames)} Masks in the directory "{pat_dir}"')
    return list(filenames)


def convert(text):
    """
    Helper function for sort_alphanumerically
    """
    if text.isdigit():
        return int(text)
    else:
        return text


def alphanum_key(key):
    """
    Helper function for sort_alphanumerically
    """
    return [convert(c) for c in re.split("([0-9]+)", key)]


def sort_alphanumerically(cts, masks):
    # Sort the lists so that we have a mapping between ct slices and masks
    cts_list = sorted(cts, key=alphanum_key)
    masks_list = sorted(masks, key=alphanum_key)
    print(cts_list[0:5])
    return cts_list, masks_list


def main():
    # # Define model
    # Uncomment the model you want to test
    model = UNet(n_channels=1, n_classes=1)
    # model = AttUNet(n_channels=1, n_classes=1)
    device = torch.device(
        "cpu"
    )  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet_model = "unet_checkpoint.pth"
    attunet_model = "attunet_checkpoint.pth"
    # Load the trained model
    # Change between unet and attunet appropriately
    model.load_state_dict(torch.load(unet_model, map_location=device))

    # # Run inference on a single image
    path = "test_mask.png"
    path2 = "test_ct.npy"
    assert os.path.exists(path) or os.path.exists(
        path2
    ), "One of the files does not exist"
    mask = Image.open(path)
    # img = Image.open(path2)
    img = np.load(path2)
    # Pass image to run inference on and then plot prediction
    pred_mask = predict_image(model, img, device, mask)

    # plot_img_and_mask(img, pred_mask, mask)
    plot_mask_on_img(img, pred_mask, mask)

    # # Find all predictions in a single patient that are not True Negative (i.e. contain annotations)
    # ct_dir = "patientCTdir"
    # mask_dir = "patientMaskdir"
    # find_non_TN(model, ct_dir, mask_dir, device)

    # # Run inference on entire set and return dice coefficient
    # data_transform_val_test = v2.Compose([
    #     # Truncate(),
    #     NormalizationMinMax(),
    #     ToTensor(),
    # ])
    # train_dataset = CTCatheterDataset(train_cts, train_masks, transform=data_transform_val_test, train=False)
    # val_dataset = CTCatheterDataset(val_cts, val_masks, transform=data_transform_val_test, train=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # #predict_dice_score(model, train_dataloader, device)
    #
    # MASK_NPZ_PATH = 'MasksV2/maskTest.npz'
    # CT_NPZ_PATH = 'CTimagesV2/ctTest.npz'
    # test_cts, test_masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
    # test_dataset = CTCatheterDataset(test_cts, test_masks, transform=data_transform_val_test, train=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # # Uncomment this only when you are sure you want to see the performance on the test set
    # predict_dice_score(model, test_dataloader, device)


if __name__ == "__main__":
    main()
    # prediction_time()
