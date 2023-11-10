from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from Metrics import dice_coeff, dice_loss
from Train import load_data_npz, split_data, show_sample, CTCatheterDataset, ToTensor, NormalizationMinMax, \
    RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

"""Uncomment the piece of code that you want to check"""

# # Test the DICE Coefficient
# # Seems to work for masks and so does the dice loss
# path = "/home/ERASMUSMC/099035/Documents/Masks/Mask0_26.png"
# path2 = "/home/ERASMUSMC/099035/Documents/Masks/Mask0_27.png"
# img = Image.open(path)
# img2 = Image.open(path2)
# img2 = np.asarray(img2)
# img2 = img2[:, :, 0]
# img2 = img2 / 255
# img2 = np.expand_dims(img2, axis=0)
# img2 = torch.from_numpy(img2)
# img_np = np.asarray(img)
# img_np = img_np[:, :, 0]
# print(img_np.shape)
# # Normalize the image first
# img_np = img_np / 255
# # Turn into tensor
# # Adds 1d on the left part
# img_t = np.expand_dims(img_np, axis=0)
# print(img_t.shape)
# img_t = torch.from_numpy(img_t)
# dice_test = dice_coeff(img_t, img2, reduce_batch_first=True)
# dice_test_loss = dice_loss(img_t, img2)
# print(f"Dice Coefficient: {dice_test}\nDice Loss: {dice_test_loss}")

# Test if the distribution of classes is equal
# MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/Masks/mask_npz.npz'
# CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimages/ct_npz.npz'
# # Load the data
# # cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
# cts, masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
# # Split the data into train/val/test
# # Train-80%,Val-10%-Test-10%
# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_data(cts, masks)
#
# def class_balance(split,split_n):
#     mask_with_annots = 0
#     mask_without_annots = 0
#     print(f'{split_n} set')
#     for item in split:
#         # If item contains non-zero items it means we have a mask with catheter points
#         if np.any(item[1]):
#             mask_with_annots += 1
#         else:
#             mask_without_annots += 1
#     mask_with_perc = mask_with_annots / len(split)
#     mask_without_perc = mask_without_annots / len(split)
#     print(f'# of Masks with annotations: {mask_with_annots} ({mask_with_perc:.2f}%)\n'
#           f'# of Masks without annotations: {mask_without_annots} ({mask_without_perc:.2f}%)')
#
# split_list = [train_y, valid_y, test_y]
# split_names = ['Train', 'Validation', 'Test']
#
# for idx, split in enumerate(split_list):
#     class_balance(split, split_names[idx])

# Test loading the datasets and dataloaders
# MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/Masks/mask_npz.npz'
# CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimages/ct_npz.npz'
# # Load the data
# # cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
# cts, masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
# # Split the data into train/val/test
# # Train-80%,Val-10%-Test-10%
# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_data(cts, masks)
# #Create pytorch dataset
# data_transform = v2.Compose([
#     # Truncate(),
#     NormalizationMinMax(),
#     ToTensor(),
#     # The following are based on the commented sections of v2
#     RandomHorizontalFlip(chance=0.5),
#     RandomVerticalFlip(chance=0.5),
#     RandomRotation(degrees=(-30, +30), chance=0.5)  # Could experiment with bilinear interpolation instead of nearest
#     #        v2.RandomHorizontalFlip(),
#     #        v2.RandomVerticalFlip(),
#     #        v2.RandomRotation(degrees = (-30,+30))
#     # Dont think Standardization or z-score normalization is good in our case
#     #        transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                             std=[0.229, 0.224, 0.225])
# ])
#
# data_transform_val_test = v2.Compose([
#     # Truncate(),
#     NormalizationMinMax(),
#     ToTensor(),
# ])
# train_dataset = CTCatheterDataset(train_x, train_y, transform=data_transform, train=True)
# val_dataset = CTCatheterDataset(valid_x, valid_y, transform=data_transform_val_test)
# test_dataset = CTCatheterDataset(test_x, test_y, transform=data_transform_val_test)
# n_train = len(train_dataset)
# n_val = len(val_dataset)
# n_test = len(test_dataset)
# print(f"Train size: {n_train}\nValidation size: {n_val}\nTest size: {n_test}")
#
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
# n_batch_train = len(train_dataloader)
# n_batch_val = len(val_dataloader)
# # Check the size of the dataset with the augmentations
# # Original size was 327 (4 data per batch)
# # Augmented size is 1635 (4 data per batch)
# print(f"Train batch size: {n_batch_train}\nValidation batch size: {n_batch_val}\nTest batch size: {len(test_dataloader)}")

# Display image and mask from dataloader
# for i_batch, sample_batched in enumerate(train_dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['mask'].size())
#     images_batch = sample_batched['image']
#     masks_batch = sample_batched['mask']
#
#     if i_batch == 3:
#         # Show only 3rd batch
#         fig, axs = plt.subplots(2, 1, figsize=(10, 6))
#         grid = utils.make_grid(images_batch)
#         axs[0].imshow(grid.numpy().transpose((1, 2, 0)))
#         grid = utils.make_grid(masks_batch)
#         axs[1].imshow(grid.numpy().transpose((1, 2, 0)))
#         fig.suptitle("Batches")
#         plt.show()
#         break

# Display sample from dataset
# fig = plt.figure()
#
# for i, sample in enumerate(train_dataset):
#     print(i, sample['image'].shape, sample['mask'].shape)
#
#     ax = plt.subplot(2, 2, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_sample(**sample)
#
#     if i == 3:
#         plt.show()
#         break


# labeled_ct, num_features= ndimage.label(cts[48][1],structure= np.ones((3,3)))
# plt.imshow(labeled_ct,cmap='gray')
# Test histogram equalization
# equ = cv2.equalizeHist(cts[13][1])
# res = np.hstack((cts[13][1],equ)) #stacking images side-by-side
# plt.imshow(res,cmap='gray')

# # Test sorting alphanumerically
# MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/Masks/mask_npz.npz'
# CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimages/ct_npz.npz'
# # Load the data
# # cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
# # Load the saved npz
# npz_cts = np.load(CT_NPZ_PATH)
# npz_masks = np.load(MASK_NPZ_PATH)
#
#
# def convert(text):
#     if text.isdigit():
#         return int(text)
#     else:
#         return text
#
#
# def alphanum_key(key):
#     return [convert(c) for c in re.split('([0-9]+)', key[0])]
#
#
# # Sort the lists so that we have a mapping between ct slices and masks
# cts_list = sorted(list(npz_cts.items()), key=alphanum_key)
# masks_list = sorted(list(npz_masks.items()), key=alphanum_key)
# # NpzFile class must be closed to avoid leaking file descriptors.
# npz_cts.close()
# npz_masks.close()
# print(cts_list[0:5])


# Test if the distribution of classes is equal in the patient wise split
MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/MasksV2/maskTrain.npz'
CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimagesV2/ctTrain.npz'
# Load the data
# cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
# Train-80%,Val-10%-Test-10%
train_cts, train_masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/MasksV2/maskVal.npz'
CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimagesV2/ctVal.npz'
val_cts, val_masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/MasksV2/maskTest.npz'
CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimagesV2/ctTest.npz'
test_cts, test_masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)

# Print the percentages of the sets
total_len = len(train_cts) + len(val_cts) + len(test_cts)
print("Ignore above as its from the Train.py file")
print(
    f'Train set contains {len(train_cts)} samples or {np.round(len(train_cts) / total_len * 100, 1)}% of the total data')
print(
    f'Validation set contains {len(val_cts)} samples or {np.round(len(val_cts) / total_len * 100, 1)}% of the total data')
print(
    f'Test set contains {len(test_cts)} samples or {np.round(len(test_cts) / total_len * 100, 1)}% of the total data')


def class_balance(split, split_n):
    mask_with_annots = 0
    mask_without_annots = 0
    print(f'{split_n} set')
    for item in split:
        # If item contains non-zero items it means we have a mask with catheter points
        if np.any(item[1]):
            mask_with_annots += 1
        else:
            mask_without_annots += 1
    mask_with_perc = mask_with_annots / len(split)
    mask_without_perc = mask_without_annots / len(split)
    print(f'# of Masks with annotations: {mask_with_annots} ({mask_with_perc:.2f}%)\n'
          f'# of Masks without annotations: {mask_without_annots} ({mask_without_perc:.2f}%)')


split_list = [train_masks, val_masks, test_masks]
split_names = ['Train', 'Validation', 'Test']

for idx, split in enumerate(split_list):
    class_balance(split, split_names[idx])
