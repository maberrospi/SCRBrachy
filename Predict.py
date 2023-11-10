# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from Metrics import multiclass_dice_coeff, dice_coeff, dice_loss
from Train import CTCatheterDataset, train_x, train_y, valid_x, valid_y, NormalizationMinMax, ToTensor
from Model import UNet


def predict_dice_score(model, dataset, device, threshold=0.5, train_img_size=(512, 512)):
    model.eval()
    dice_score = 0
    for batch in tqdm(dataset, desc='Image'):
        image, true_mask = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = model(image).cpu()
            # Down/up sample the input image to the required size if it's not
            output = F.interpolate(output, train_img_size, mode='bilinear')
            mask = torch.sigmoid(output) > threshold
            dice_score += dice_coeff(mask, true_mask, reduce_batch_first=False)
    dice_score = dice_score / len(dataset)
    print(f'Final dice score: {dice_score}')
    return dice_score


def predict_image(model, img, device, orig, threshold=0.5, train_img_size=(512, 512)):
    model.eval()
    image = CTCatheterDataset.preprocess(img)
    # Unsqueeze as the model expects BxCxHxW
    image = torch.unsqueeze(image, 0)
    img = image.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        # Move all model parameters and buffers to the cpu
        output = model(img).cpu()
        orig = np.asarray(orig)
        orig = orig[:, :, 0]
        orig = orig / 255
        orig = np.expand_dims(orig, axis=0)
        orig = torch.from_numpy(orig)
        output1 = torch.sigmoid(output) > threshold
        print(dice_coeff(torch.squeeze(output1, dim=0), orig))
        # Down/up sample the input image to the required size if it's not
        output = F.interpolate(output, train_img_size, mode='bilinear')
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
    axs[0].imshow(image, cmap='gray')
    axs[1].imshow(orig_mask, cmap='gray')
    axs[2].imshow(mask, cmap='gray')
    axs[0].set_title("CT slice")
    axs[1].set_title("Original mask")
    axs[2].set_title("Predicted mask")
    fig.tight_layout()
    plt.show()


def main():
    # Define model
    model = UNet(n_channels=1, n_classes=1)
    device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the trained model
    model.load_state_dict(
        torch.load("/home/ERASMUSMC/099035/Desktop/PythonWork/Baseenv/checkpoints/debugging4/checkpoint_epoch10.pth",
                   map_location=device))

    # # Run inference on a single image
    path = "/home/ERASMUSMC/099035/Documents/Masks/Mask11_34.png"
    path2 = "/home/ERASMUSMC/099035/Documents/CTimages/CT11_34.png"
    mask = Image.open(path)
    img = Image.open(path2)
    # Pass image to run inference on and then plot prediction
    pred_mask = predict_image(model, img, device, mask)
    plot_img_and_mask(img, pred_mask, mask)

    # Run inference on entire set and return dice coefficient
    # data_transform_val_test = v2.Compose([
    #     # Truncate(),
    #     NormalizationMinMax(),
    #     ToTensor(),
    # ])
    # train_dataset = CTCatheterDataset(train_x, train_y, transform=data_transform_val_test, train=False)
    # val_dataset = CTCatheterDataset(valid_x, valid_y, transform=data_transform_val_test, train=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # predict_dice_score(model, train_dataloader, device)


if __name__ == "__main__":
    main()
