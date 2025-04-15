import random
from typing import (
    Dict, 
    Tuple, 
    List
)
from copy import deepcopy
import numpy as np
import torch
from torch import nn, Tensor
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns



def reject_randomness(manualSeed):
    np.random.seed(manualSeed)
    random.rand.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None



def sum_model_parameters(model: nn.Module) -> int:
    """
    Sum up the number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def find_shapes_for_swivels(
    model: nn.Module, 
    swivels: List[str],
    input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256)
) -> Dict[str, List[int]]:
    # Create a dictionary to store the output shapes
    model = deepcopy(model).cuda()
    output_shapes = {}

    # Get hook function to capture and print output shapes for swivel
    def get_hook_fn(name):
        def hook_fn(module, input, output):
            output_shapes[name] = list(output.shape)
        return hook_fn

    # Attach hooks to all layers
    hooks = []
    for layer_id in swivels:
        layer   = model.get_submodule(layer_id)
        hook_fn = get_hook_fn(layer_id)
        hook    = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Run a sample input through the model
    x = torch.randn(input_shape).cuda()  # Batch size 1, 3 channels, 32x32 image
    _ = model(x)

    # remove hooks and model
    for hook in hooks:
        hook.remove()

    del model

    return output_shapes



def eAURC(predicted_risks, true_risks, ret_curves=False):
    n = len(true_risks)
    true_risks_sorted, _ = true_risks.sort()
    _, predicted_indices = predicted_risks.sort()
    true_risks_aggr = true_risks_sorted.cumsum(0) / torch.arange(1, n + 1)
    aurc_opt = true_risks_aggr.mean()
    predicted_risks_aggr = true_risks[predicted_indices].cumsum(0) / torch.arange(1, n + 1)
    aurc_pred = predicted_risks_aggr.mean()
    assert true_risks_aggr[-1] == predicted_risks_aggr[-1]
    if ret_curves:
        return aurc_pred - aurc_opt, true_risks_aggr, predicted_risks_aggr
    else:
        return aurc_pred - aurc_opt



def collect_eval_from_predictions(
    predictions: Dict, 
):
    evaluation = {}
    
            
    # for best predictors (train, val), collect results and save
    for key, value in predictions.items():
        if key == 'swivels':
            continue

        # init evaluation dictionary for this dataset 
        eval_dict = {}

        # metrics to track
        eval_dict['corr']            = []
        eval_dict['mae']             = []
        eval_dict['predicted_risks'] = []
        eval_dict['original_risks']  = []
        eval_dict['predictor_idx']   = []
        eval_dict['swivel']          = []

        for i in range(len(predictions['swivels'])):

            original_risks = 1 - value['true_score'].squeeze(1)
            predicted_risk = 1 - value['predicted_score'][:, i]
            eval_dict['corr'].append(torch.corrcoef(torch.stack([predicted_risk, original_risks], dim=0))[0, 1])
            eval_dict['mae'].append((predicted_risk - original_risks).abs().mean().item())
            eval_dict['predicted_risks'].append(predicted_risk)
            eval_dict['original_risks'].append(original_risks)
            eval_dict['predictor_idx'].append(i)
            eval_dict['swivel'].append(predictions['swivels'][i])

        evaluation[key] = eval_dict

    return evaluation



def clean_predictions(
    predictions: Dict, 
    datamodule: L.LightningDataModule, 
    model: L.LightningModule
):

    predictions_clean = {d:
        {
            key: torch.cat([d[key] for d in m], dim=0) 
            for key in m[0].keys()
        } for d, m in zip(datamodule.test_dataloader().keys(), predictions)
    }

    predictions_clean['swivels'] = [adapter.swivel for adapter in model.wrapper.adapters]

    # sanity check for data matching
    for key, value in predictions_clean.items():
        if key == 'swivels':
            continue
        dataset_length = datamodule.test_dataloader()[key].dataset.__len__()
        assert value['true_score'].shape[0] == dataset_length, f'{key} dataset length mismatch'

    return predictions_clean



def get_risk_coverage_curves(dicts, sources):
    """
    Plot risk-coverage curves for multiple dictionaries.

    Args:
        dicts (list of dict): List of nested dictionaries containing 'risks' and 'weights'.
        sources (list of str): List of source names corresponding to each dictionary.

    Returns:
        Matplotlib figure
    """
    # Extract datasets
    datasets = list(dicts[0].keys())  # Assume all dicts have the same top-level keys

    # Create subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    sns.set(style="whitegrid")

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        for source, data_dict in zip(sources, dicts):
            # Extract risks and weights
            risks = data_dict[dataset]['risks'][0]
            weights = data_dict[dataset]['weights'][0]
            # Convert to numpy for plotting
            risks = risks.numpy() if isinstance(risks, Tensor) else risks
            weights = weights.numpy() if isinstance(weights, Tensor) else weights
            # Plot the curve
            ax.plot(weights, risks, label=source)

        # Add ideal line if present
        ideal_risks = data_dict[dataset].get('ideal_risks')
        ideal_weights = torch.linspace(0, 1, steps=len(ideal_risks))
        ax.plot(ideal_weights, ideal_risks, label='ideal', linestyle='--')

        ax.set_title(f"Risk-Coverage Curve: {dataset}")
        ax.set_xlabel("Coverage (Weights)")
        ax.set_ylabel("Risk")
        ax.legend(title="Source")

    plt.tight_layout()
    return fig