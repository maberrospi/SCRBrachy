#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:31:14 2023

@author: ERASMUSMC+099035
# The augmentation of data was inspired by https://discuss.pytorch.org/t/transform-and-image-data-augmentation/71942/6
Additional resources include but are not limited to:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

The training, evaluation, metrics and model definition codes were
heavily inspired by the work of https://github.com/milesial/Pytorch-UNet/tree/master
Additional resources include but are not limited to:
https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
https://debuggercafe.com/unet-from-scratch-using-pytorch/
https://www.geeksforgeeks.org/u-net-architecture-explained/
https://doi.org/10.1007/978-3-319-24574-4_28
https://doi.org/10.5114%2Fjcb.2021.106118
https://doi.org/10.1016/j.brachy.2019.06.003
"""

# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import ndimage
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torchinfo import summary
import random
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from Model import UNet
from Metrics import dice_loss, dice_coeff
from Evaluate import evaluate, evaluate_loss


# %% DEFINE ALL FUNCTIONS HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_data_npz(mask_path, cts_path):
    # Load the saved npz
    npz_cts = np.load(cts_path)
    npz_masks = np.load(mask_path)
    # Sort the lists so that we have a mapping between ct slices and masks
    cts_list = sorted(list(npz_cts.items()))
    masks_list = sorted(list(npz_masks.items()))
    # NpzFile class must be closed to avoid leaking file descriptors.
    npz_cts.close()
    npz_masks.close()
    return cts_list, masks_list


def split_data(cts, masks):
    # The data is shuffled by the function
    train_perc, val_perc, test_perc = 0.8, 0.1, 0.1
    # Get validation set
    train_x, valid_x, train_y, valid_y = train_test_split(cts, masks, test_size=val_perc, random_state=42)
    # train_y, valid_y = train_test_split(masks, test_size=val_perc, random_state=42)
    # Get training and test sets
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_perc / (1 - val_perc),
                                                        random_state=42)
    # train_y, test_y = train_test_split(train_y, test_size=test_perc/(1-val_perc), random_state=42)
    # Print the percentages of the sets
    print(
        f'Train set contains {len(train_x)} samples or {np.round(len(train_x) / len(cts) * 100, 1)}% of the total data')
    print(
        f'Train set contains {len(valid_x)} samples or {np.round(len(valid_x) / len(cts) * 100, 1)}% of the total data')
    print(
        f'Train set contains {len(test_x)} samples or {np.round(len(test_x) / len(cts) * 100, 1)}% of the total data')
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def view_centered(cts, masks, ct_number):
    """View the centered image {Mask and CT} based on the calculation of the center of mass from the CT slice"""
    # labeled_ct, num_features= ndimage.label(cts[ct_number][1], structure= np.ones((3,3)))
    # Add padding on all edges to test a theory
    # padded_ct = np.pad(cts[ct_number][1],100,'constant',constant_values = 0)
    # Calculate center of mass
    # com = measure.centroid(cts[ct_number][1])
    com = ndimage.center_of_mass(cts[ct_number][1])
    # Round the values returned and change type to uint8
    com = np.round(com).astype(np.uint8)
    print(com)
    # Define truncated image size
    imsize = int(192 / 2)
    # Plot the original CT slice and the truncated CT slice
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    axs[0].imshow(cts[ct_number][1][com[0] - imsize:com[0] + imsize, com[1] - imsize:com[1] + imsize], cmap='gray')
    axs[0].set_title("Centered at center of mass")
    axs[0].set_xticks(np.arange(0, imsize * 2, step=32))
    axs[0].set_yticks(np.arange(0, imsize * 2, step=32))
    axs[1].set_title("Original")
    axs[1].imshow(cts[ct_number][1], cmap='gray')
    axs[1].set_xticks(np.arange(0, cts[ct_number][1].shape[0], step=100))
    axs[1].set_yticks(np.arange(0, cts[ct_number][1].shape[1], step=100))
    axs[1].plot(com[1], com[0], 'r+')
    axs[2].imshow(masks[ct_number][1][com[0] - imsize:com[0] + imsize, com[1] - imsize:com[1] + imsize], cmap='gray')
    axs[2].set_title("Mask centered")
    axs[2].set_xticks(np.arange(0, imsize * 2, step=32))
    axs[2].set_yticks(np.arange(0, imsize * 2, step=32))


class CTCatheterDataset(Dataset):
    """CTs and Catheter Masks Dataset"""

    def __init__(self, cts, masks, transform=None, train=False):
        """
        Arguments:
            cts (List): CT list containing a tuple of size 2 where index0 is the the name and index1 is the numpy array
            masks (List): Mask list containing a tuple of size 2 where index0 is the the name and index1 is the numpy array
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cts = cts
        self.masks = masks
        self.transform = transform
        self.train = train

    def __len__(self):
        # Returns the count of samples
        # Return 5 times the original size to account for augmentations if train is true
        if self.train:
            return 5 * len(self.cts)
        else:
            return len(self.cts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # account for augmentations if train is true
        if self.train:
            idx = idx % len(self.cts)

        image = self.cts[idx][1]
        mask = self.masks[idx][1]
        sample = {'image': image, 'mask': mask}

        # Set RNG for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)  # Just in case although not used I believe

        if self.transform:
            sample = self.transform(sample)
            # sample = self.transform(sample['image'],sample['mask'])

        return sample

    @staticmethod
    def preprocess(img):
        # Turn from PIL img to numpy
        img = np.asarray(img)
        # Normalize the image first
        img = img / 255
        # Images were saves as RGBA so we need to extract only one channel
        img = img[:, :, 0]
        # Turn into tensor
        # Adds 1d on the left part
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        return img


class Truncate(object):
    """ Truncate the images and masks to 192x192
        centered at the center of mass of the image
    """

    def __init__(self, output_size=192):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # Calculate center of mass
        com = ndimage.center_of_mass(image)
        # com = measure.centroid(image)
        # Round the values returned and change type to uint8
        com = np.round(com).astype(np.uint8)
        # Define truncated image size divided by 2
        halfimsize = int(self.output_size[0] / 2)
        print(com[0])
        image = image[com[0] - halfimsize:com[0] + halfimsize,
                com[1] - halfimsize:com[1] + halfimsize]

        mask = mask[com[0] - halfimsize:com[0] + halfimsize,
               com[1] - halfimsize:com[1] + halfimsize]

        print(com[0])

        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # Adds 1d on the left part
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class NormalizationMinMax(object):
    """Normalize all the data using min max normalization between 0-255"""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image / 255
        mask = mask / 255
        return {'image': image,
                'mask': mask}


class RandomHorizontalFlip(object):
    """Perform random horizontal flip"""

    def __init__(self, chance=0.5):
        assert isinstance(chance, float)
        assert (0 <= chance <= 1)
        self.chance = chance

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Random horizontal flipping
        if random.random() > self.chance:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return {'image': image,
                'mask': mask}


class RandomVerticalFlip(object):
    """Perform random horizontal flip"""

    def __init__(self, chance=0.5):
        assert isinstance(chance, float)
        assert (0 <= chance <= 1)
        self.chance = chance

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Random horizontal flipping
        if random.random() > self.chance:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return {'image': image,
                'mask': mask}


class RandomRotation(v2.RandomRotation):
    """Perform random rotations"""

    def __init__(self, degrees, chance=0.5):
        # Initialize the super class with degrees as passed in the input
        super().__init__(degrees=degrees, interpolation=TF.InterpolationMode.NEAREST)
        assert isinstance(chance, float)
        assert (0 <= chance <= 1)
        self.chance = chance

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if random.random() > self.chance:
            params = super()._get_params(self)
            image_new = super()._transform(image, params)
            mask_new = super()._transform(mask, params)

            return {'image': image_new,
                    'mask': mask_new}
        else:
            return {'image': image,
                    'mask': mask}


# Inspiration https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# and https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# Can maybe def __call__ later on to make ti cleaner
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float("inf")
        self.counter = 0

    def early_stop(self, val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False


def train_model(
        model,
        device,
        batch_size=4,
        learning_rate=1e-5,
        epochs=5,
        save_checkpoint=True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        bilinear_upsampling=False,
        # Fast and memory efficient training
        amp=True,

):
    # Get a summary of the model and show it
    mod_sum = summary(model,
                      input_size=(batch_size, 1, 512, 512),
                      # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
                      verbose=0,
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20,
                      row_settings=["var_names"]
                      )
    print(mod_sum)

    # Create a writer with all default settings for use with TensorBoard
    writer = SummaryWriter(log_dir='/home/ERASMUSMC/099035/Desktop/PythonWork/Baseenv/runs/debugging4batch32')

    # 1. Create datasets 
    # We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
    # some APIs may slightly change in the future
    torchvision.disable_beta_transforms_warning()

    data_transform = v2.Compose([
        # Truncate(),
        NormalizationMinMax(),
        ToTensor(),
        RandomHorizontalFlip(chance=0.5),
        RandomVerticalFlip(chance=0.5),
        RandomRotation(degrees=(-30, +30), chance=0.5)
        # Could experiment with bilinear interpolation instead of nearest
    ])

    data_transform_val_test = v2.Compose([
        # Truncate(),
        NormalizationMinMax(),
        ToTensor(),
    ])

    # Limit the train and validation sets to make it easier to debug
    # train_x_t = train_x[0:50]
    # valid_x_t = valid_x[0:10]
    # train_y_t = train_y[0:50]
    # valid_y_t = valid_y[0:10]
    # train_dataset = CTCatheterDataset(train_x_t, train_y_t, transform=data_transform, train=False)
    # val_dataset = CTCatheterDataset(valid_x_t, valid_y_t, transform=data_transform, train=False)
    train_dataset = CTCatheterDataset(train_x, train_y, transform=data_transform, train=True)
    val_dataset = CTCatheterDataset(valid_x, valid_y, transform=data_transform_val_test, train=False)
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"Train size: {n_train}\nValidation size: {n_val}")

    # 2. Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    n_batch_train = len(train_dataloader)
    n_batch_val = len(val_dataloader)
    # Check the size of the dataset with the augmentations
    # Original size was 327 (4 data per batch)
    # Augmented size is 1635 (4 data per batch)
    print(f"Train batch size: {n_batch_train}\nValidation batch size: {n_batch_val}")

    # Initialize Logging
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
        ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # BCEWithLogitsLoss takes prediction as raw input instead of having to wrap it in sigmoid function
    # https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
    # Test using pos_weight
    pos_weight = torch.full([1, 512, 512], 100)
    pos_weight = pos_weight.to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()  # pos_weight=pos_weight)
    early_stopper = EarlyStopping(patience=5, min_delta=0)
    global_step = 0

    # 4. Begin Training
    for epoch in range(1, epochs + 1):
        # Make sure gradient tracking is on
        model.train()
        # Set epoch loss back to 0
        epoch_train_loss = 0
        epoch_train_score = 0
        train_score = 0
        epoch_val_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_dataloader:
                images, true_masks = batch['image'], batch['mask']
                # Throw error if n_channels is not correct
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # Move images and labels to correct device and type
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # Make sure the input data does not have any NaN values
                assert not torch.isnan(images).any()
                assert not torch.isnan(true_masks).any()
                # This does not allow mps which is for GPU use in Mac devices
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # Run forward pass - Make prediction on images
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # I have a feeling that the mask_pred squeezed and the true masks will NOT have the same size
                        # Indeed it had an error and i added squeeze to both to get a size of BxHxW
                        loss = criterion(masks_pred, true_masks.float())
                        # print(loss)
                        loss += dice_loss(F.sigmoid(masks_pred), true_masks.float(),
                                          multiclass=False)
                        train_score = dice_coeff((F.sigmoid(masks_pred) > 0.5).float(), true_masks.float())
                    else:
                        # Pretty sure this needs to be fixed but doesnt affect me now
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                # Zero your gradients for every batch before performing backpropagation
                # See zero_grad docs for why set_to_none
                optimizer.zero_grad(set_to_none=True)
                # Scale the gradients to prevent underflow's when the gradient has a small magnitude
                # and cannot be represented with float16
                # And then perform backpropagation
                grad_scaler.scale(loss).backward()
                # Clip the gradients to mitigate the problem of exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # The step function invokes unscale to unscale the previously scaled gradients
                # And then invoked optimizer.step() -> Performs single optimization step and parameter update
                grad_scaler.step(optimizer)
                # Update the scale factor
                grad_scaler.update()

                # update tqdm bar
                pbar.update(images.shape[0])
                global_step += 1
                epoch_train_loss += loss.item()
                epoch_train_score += train_score
                # Display the training loss
                pbar.set_postfix(**{'Train loss (batch)': loss.item()})
                # The evaluation is performed at a fixed number of training steps even though the general practice
                # is to perform evaluation after every epoch but this is acceptable in an academic context.
                # https://stackoverflow.com/questions/61024500/when-to-do-validation-when-we-are-training-based-off-of-training-steps
                # Perform evaluation on validation data
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_dataloader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))

        # Steps for computing the validation loss after every epoch
        # I have a feeling that the mask_pred squeezed and the true masks will NOT have the same size
        epoch_val_loss = evaluate_loss(model, val_dataloader, criterion, device, amp)
        # Compute epoch train loss
        epoch_train_loss = epoch_train_loss / n_batch_train
        # Compute epoch train score
        epoch_train_score = epoch_train_score / n_batch_train
        logging.info(
            f'Validation Dice loss: {epoch_val_loss}\nTrain Dice loss: {epoch_train_loss}\nTrain score: {epoch_train_score}')

        # Log the average dice loss per epoch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': epoch_train_loss, 'Validation': epoch_val_loss},
                           epoch + 1)
        writer.flush()

        # Might not be needed to save the mask_values, im positive I don't need it in my case
        if save_checkpoint:
            Path(DIR_CHECKPOINT).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = train_dataset.masks[:, 1]
            torch.save(state_dict, str(DIR_CHECKPOINT / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if early_stopper.early_stop(epoch_val_loss):
            logging.info(f'Early stop of training at epoch {epoch}')
            break


def show_sample(image, mask):
    # This results in rotated images
    image = torch.transpose(torch.squeeze(image), 0, 1)
    mask = torch.transpose(torch.squeeze(mask), 0, 1)
    stack = np.hstack((image, mask))
    plt.imshow(stack, cmap='gray')
    plt.pause(0.001)  # pause a bit so that plots are updated


# %% Global Variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MASK_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/Masks/mask_npz.npz'
CT_NPZ_PATH = '/home/ERASMUSMC/099035/Documents/CTimages/ct_npz.npz'
DIR_CHECKPOINT = Path('/home/ERASMUSMC/099035/Desktop/PythonWork/Baseenv/checkpoints/debugging4/')

# Load the data
# cts and masks are lists of tuples where tuple index0 is the name and index1 is the numpy array
cts, masks = load_data_npz(MASK_NPZ_PATH, CT_NPZ_PATH)
# Split the data into train/val/test
# Train-80%,Val-10%-Test-10%
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_data(cts, masks)


# As a reminder, sets have a shape of HxW


# %% Main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    # View an example of a centered CT and mask
    # view_centered(cts,masks,13)
    # view_centered(train_x,train_y,41)

    # We are using BETA APIs, so we deactivate the associated warning, thereby acknowledging that
    # some APIs may slightly change in the future
    torchvision.disable_beta_transforms_warning()

    # Hypeparameters
    batch_size = 32
    learning_rate = 1e-6
    epochs = 10
    save_checkpoint = True
    weight_decay: float = 1e-8
    momentum: float = 0.999
    gradient_clipping: float = 1.0
    bilinear_upsampling = False
    # Fast and memory efficient training
    amp = True

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=1, bilinear=bilinear_upsampling)
    # Tensor is or will be allocated in dense non-overlapping memory.
    # Strides represented by values in strides[0] > strides[2] > strides[3] > strides[1] == 1 aka NHWC order.
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Move the model to the current device (CPU or GPU)
    model.to(device=device)

    try:
        train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_checkpoint=save_checkpoint,
            device=device,
            amp=amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            amp=amp
        )


if __name__ == '__main__':
    main()
# Inspired by https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7#:~:text=Num_workers%20tells%20the%20data%20loader,the%20GPU%20has%20to%20wait.
# Will be useful to check after I have determined that the training code works correctly
# to make the process faster
# Could also be beneficial to use drop_last to drop the last batch if its smaller than batch size
# from time import time
# import multiprocessing as mp
# for num_workers in range(2, mp.cpu_count(), 2):
#    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=4,pin_memory=True)
#    start = time()
#    for epoch in range(1, 3):
#        for i, data in enumerate(train_loader, 0):
#            pass
#    end = time()
#    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))