#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:39:58 2023

@author: ERASMUSMC+099035
"""

import torch
from torch import Tensor


def dice_coeff(inp: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6) -> float:
    """
    Calculates the DSC of two inputs
    @param inp: First input tensor of 2,3 or 4 dimensions
    @param target: Second input tensor of 2,3 or 4 dimensions
    @param reduce_batch_first: Boolean of whether the batch dimension has been removed
    @param epsilon: Smoothing factor
    @return: Float Mean DSC
    """
    # Calculate average of Dice coefficient for all batches, or for a single mask
    assert inp.size() == target.size()
    # If input is CxHxW or reduce batch is false pass the test
    assert inp.dim() == 3 or not reduce_batch_first
    # Sum dimensions are WxH if input is HxW or if reduce batch is false
    sum_dim = (-1, -2) if inp.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    intersection = 2 * (inp * target).sum(dim=sum_dim)
    sets_sum = inp.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, intersection, sets_sum)

    dice = (intersection + epsilon) / (sets_sum + epsilon)
    return dice.mean()


# This function was not used for the purpose of this study but could be useful
def multiclass_dice_coeff(inp: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculates mean DSC for multiple classes
    """
    # Average of Dice coefficient for all classes
    return dice_coeff(inp.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(inp: Tensor, target: Tensor, multiclass: bool = False):
    """
    Calculated the DSC loss which is 1-DSC
    @param inp: First input tensor of 2,3 or 4 dimensions
    @param target: Second input tensor of 2,3 or 4 dimensions
    @param multiclass: Boolean to use multiclass or single class DSC function
    @return: Float DSC loss
    """
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inp, target, reduce_batch_first=False)
