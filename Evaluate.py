# %% Import libraries  HERE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn.functional as F
from tqdm import tqdm
from Metrics import multiclass_dice_coeff, dice_coeff, dice_loss


# Run in inference mode which disables grads
@torch.inference_mode()
def evaluate(model, validation_loader, device, amp):
    """
    Evaluate the current validation DSC of the model in training

    @param model: Model being trained
    @param validation_loader: DataLoader instance of the validation set
    @param device: CPU or GPU
    @param amp: Boolean to enable/disable amp (Automatic Mixed Precision)
    @return: Mean DSC of the validation set
    """
    # Set model to evaluation mode
    model.eval()
    num_val_batches = len(validation_loader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(validation_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Make a prediction
            mask_pred = model(images)

            if model.n_classes == 1:
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                # Steps for computing the DICE score
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # Compute the DICE score
                dice_score += dice_coeff(mask_pred, true_masks.float(), False)
            else:
                assert true_masks.min() >= 0 and true_masks.max() <= model.n_classes, 'True mask indices should be in [' \
                                                                                      '0, n_classes] '
                # convert to one-hot format
                true_masks = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], true_masks[:, 1:], False)
        # Set model back to training mode
        model.train()
        return dice_score / max(num_val_batches, 1)


# Run in inference mode which disables grads
@torch.inference_mode()
def evaluate_loss(model, validation_loader, criterion, device, amp):
    """
    Evaluate the current validation DSC loss of the model in training

    @param model: Model being trained
    @param validation_loader: DataLoader instance of the validation set
    @param criterion: Loss criterion (for example BCEWithLogitsLoss)
    @param device: CPU or GPU
    @param amp: Boolean to enable/disable amp (Automatic Mixed Precision)
    @return: Mean DSC loss of the validation set
    """
    # Set model to evaluation mode
    model.eval()
    num_val_batches = len(validation_loader)
    val_loss = 0
    total_val_loss = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(validation_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Make a prediction
            mask_pred = model(images)

            if model.n_classes == 1:
                assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
                val_loss = criterion(mask_pred, true_masks.float())
                val_loss += dice_loss(F.sigmoid(mask_pred), true_masks.float(), multiclass=False)
            else:
                assert true_masks.min() >= 0 and true_masks.max() <= model.n_classes, 'True mask indices should be in [' \
                                                                                      '0, n_classes] '
                val_loss = criterion(mask_pred, true_masks)
                val_loss += dice_loss(
                    F.softmax(mask_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )

            total_val_loss += val_loss

    return total_val_loss / num_val_batches
