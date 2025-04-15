from typing import Callable
from torch import (
    Tensor,
    nn,
    isnan,
    where,
    tensor,
)
from torch.nn.functional import one_hot
from monai.metrics import (
    DiceMetric, 
    SurfaceDiceMetric
)



def dice_per_class_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    criterion: Callable = nn.MSELoss(reduction='none'),
    num_classes: int = 4,
    return_scores: bool = False
) -> Tensor:
    """
    Calculate the Dice coefficient per class loss.

    Args:
        predicted_segmentation (Tensor): The predicted segmentation tensor.
        target_segmentation (Tensor): The target segmentation tensor.
        prediction (Tensor): The prediction tensor.
        criterion (Callable, optional): The loss function to use. Defaults to nn.MSELoss(reduction='none').
        num_classes (int, optional): The number of classes. Defaults to 4.
        return_scores (bool, optional): Whether to return the prediction and target tensor. Defaults to False.

    Returns:
        Tensor: The calculated loss. If return_scores is True, also returns the score tensor.
    """
    score = DiceMetric(
        include_background=True, 
        reduction="none",
        num_classes=num_classes,
        ignore_empty=False
    )(predicted_segmentation, target_segmentation).detach()
    
    not_nans = ~isnan(score) * 1.0
    not_nans = not_nans.unsqueeze(1).repeat(1, prediction.shape[1], 1)
    score = score.nan_to_num(0).detach().unsqueeze(1).repeat(1, prediction.shape[1], 1).clamp(0, 512)
    
    loss = criterion(prediction, score) * not_nans
    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    if return_scores:
        nan_mask   = where(not_nans == 0, tensor(float('nan')), not_nans)
        prediction = prediction.detach() * nan_mask
        score      = score * nan_mask
        prediction = prediction[..., 1:].nanmean(-1)
        score      = score[..., 1:].nanmean(-1)
        return loss, prediction.nan_to_num(0), score.nan_to_num(0)
    else:
        return loss

    


def surface_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    criterion: Callable = nn.MSELoss(reduction='none'),
    num_classes: int = None,
    return_scores: bool = False
) -> Tensor:
    """
    Calculate the transformed Hausdorff distance per class loss.

    Args:
        predicted_segmentation (Tensor): The predicted segmentation tensor.
        target_segmentation (Tensor): The target segmentation tensor.
        prediction (Tensor): The prediction tensor.
        criterion (Callable, optional): The loss function to use. Defaults to nn.MSELoss(reduction='none').
        num_classes (int, optional): The number of classes. Defaults to 4.
        sigma (Float, optional): Sigma for RBF transformation. Defaults to 1.0.
        return_scores (bool, optional): Whether to return the prediction and target tensor. Defaults to False.

    Returns:
        Tensor: The calculated loss. If return_scores is True, also returns the score tensor.
    """
    predicted_segmentation = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
    target_segmentation = one_hot(target_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)

    score = SurfaceDiceMetric(
        include_background=True, 
        reduction="none",
        class_thresholds=[3] * num_classes,
    )(predicted_segmentation, target_segmentation).detach()

    not_nans = ~isnan(score) * 1.0
    not_nans = not_nans.unsqueeze(1).repeat(1, prediction.shape[1], 1).detach()
    score    = score.nan_to_num(0).unsqueeze(1).repeat(1, prediction.shape[1], 1).clamp(0, 512)
    loss     = criterion(prediction, score) * not_nans
    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    if return_scores:
        nan_mask   = where(not_nans == 0, tensor(float('nan')), not_nans)
        prediction = prediction.detach() * nan_mask
        score      = score * nan_mask
        prediction = prediction[..., 1:].nanmean(-1)
        score      = score[..., 1:].nanmean(-1)
        return loss, prediction.nan_to_num(0), score.nan_to_num(0)
    else:
        return loss

