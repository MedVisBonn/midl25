from typing import (
    Callable, 
    Tuple,
    Dict,
    Union
)
from torch import (
    nn, 
    Tensor, 
    argmax, 
    stack, 
    flatten, 
    cat,
    tensor,
)
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from copy import deepcopy
from omegaconf import OmegaConf
import wandb
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
import sys

sys.path.append('../')
from utils import find_shapes_for_swivels, eAURC
from losses import dice_per_class_loss, surface_loss



def get_score_prediction_module_trainer(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    trainer_cfg: OmegaConf
):
    # infered variables
    now = datetime.now()
    # TODO
    filename = f'{data_cfg.dataset}_{data_cfg.domain}_{model_cfg.loss_fn}'
    if model_cfg.adversarial_training:
        filename += '_adversarial'
    else:
        filename += '_normal'
    if trainer_cfg.finetune:
        filename += '_finetune'
    if model_cfg.name != 'None':
        filename += f'_{model_cfg.name}'
    filename += f'_{now.strftime("%Y-%m-%d-%H-%M")}'

    # init logger
    if trainer_cfg.logging:
        wandb.finish()
        logger = WandbLogger(
            project="MIDL2025", 
            log_model=True, 
            name=filename
        )
    else:
        logger = None

    # get checkpoint path
    dirpath = trainer_cfg.root_dir + trainer_cfg.model_checkpoint.dirpath
    # return trainer
    return L.Trainer(
        limit_train_batches=trainer_cfg.limit_train_batches,
        max_epochs=trainer_cfg.max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=dirpath,
                filename=filename,
                save_top_k=trainer_cfg.model_checkpoint.save_top_k,
                mode=trainer_cfg.model_checkpoint.mode,
                monitor=trainer_cfg.model_checkpoint.monitor,
            )
        ],
        precision='16-mixed',
        gradient_clip_val=0.5,
        devices=[0],
        limit_test_batches=50
    )



def get_score_prediction_module(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    unet: nn.Module,
    metadata: Dict,
    ckpt: Union[str, None] = None
):
    ### derived variables
    if model_cfg.swivels in ['best', 'all']:
        swivels = [layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]]
    elif model_cfg.swivels == 'ConfidNet':
        swivels = ['model.2.0.adn.A']
    else:
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][model_cfg.swivels]]

    
    # init prediction head
    output_shapes = find_shapes_for_swivels(
        model=unet, 
        swivels=swivels, 
        input_shape=OmegaConf.to_object(data_cfg.input_shape)
    )

    # init detectors with corresponding predictors
    score_detectors = [
        ScorePredictionAdapter(
            swivel=swivel,
            predictor=ScorePredictor(
                input_size=output_shapes[swivel],
                output_dim=model_cfg.adapter_output_dim
            ),
            adversarial_training = model_cfg.adversarial_training,
            adversarial_prob=model_cfg.adversarial_prob
        ) for swivel in swivels
    ]
    # wrap the model with the detectors
    wrapper = ScorePredictionWrapper(
        model=unet,
        adapters=nn.ModuleList(score_detectors),
        copy=True
    )
    wrapper.freeze_normalization_layers()
    wrapper.freeze_model()

    loss_fn_dict = {
        'dice': dice_per_class_loss,
        'surface': surface_loss
    }

    if ckpt is None:
        return ScorePredictionWrapperLightningModule(
            wrapper=wrapper,
            loss_fn=loss_fn_dict[model_cfg.loss_fn],
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            adversarial_training=model_cfg.adversarial_training,
            adversarial_prob=model_cfg.adversarial_prob,
            adversarial_step_size=model_cfg.adversarial_step_size,
            num_classes=model_cfg.num_classes,
            metadata=metadata
        )

    elif isinstance(ckpt, str):
        return ScorePredictionWrapperLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt,
            wrapper=wrapper,
            loss_fn=loss_fn_dict[model_cfg.loss_fn],
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            adversarial_training=model_cfg.adversarial_training,
            adversarial_prob=model_cfg.adversarial_prob,
            adversarial_step_size=model_cfg.adversarial_step_size,
            num_classes=model_cfg.num_classes,
            metadata=metadata
        )
    
      
def get_score_prediction_finetune_module(
    wrapper: nn.Module,
    oracle: nn.Module,
    model_cfg: OmegaConf,
    metadata: Dict,
    ckpt: Union[str, None] = None
):


    for param in wrapper.model.parameters():
        param.requires_grad = True

    loss_fn_dict = {
        'dice': dice_per_class_loss,
        'surface': surface_loss
    }

    if ckpt is None:
        return ScorePredictionFinetunerLightningModule(
            wrapper=wrapper,
            oracle=oracle,
            loss_fn=loss_fn_dict[model_cfg.loss_fn],
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            adversarial_training=model_cfg.adversarial_training,
            adversarial_prob=model_cfg.adversarial_prob,
            adversarial_step_size=model_cfg.adversarial_step_size,
            num_classes=model_cfg.num_classes,
            metadata=metadata
        )

    elif isinstance(ckpt, str):
        return ScorePredictionFinetunerLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt,
            wrapper=wrapper,
            oracle=oracle,
            loss_fn=loss_fn_dict[model_cfg.loss_fn],
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            adversarial_training=model_cfg.adversarial_training,
            adversarial_prob=model_cfg.adversarial_prob,
            adversarial_step_size=model_cfg.adversarial_step_size,
            num_classes=model_cfg.num_classes,
            metadata=metadata
        )
    
 
class PredictionHead(nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim
        ):
        super(PredictionHead, self).__init__()
        hidden_dim = input_dim // 4
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Define Layer Normalization for the first layer
        self.ln1 = nn.LayerNorm(hidden_dim)
        # Leaky ReLU activation function for the first layer
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Define Layer Normalization for the second layer
        self.ln2 = nn.LayerNorm(hidden_dim)
        # Leaky ReLU activation function for the second layer
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        
        # Define the third fully connected layer (output layer)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the first layer, normalization, and activation
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.leaky_relu1(x)
        
        # Forward pass through the second layer, normalization, and activation
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.leaky_relu2(x)
        
        # Output layer (no activation here if it's a regression task or depending on the task)
        x = self.fc3(x)
        
        return x



class ScorePredictor(nn.Module):
    """
    A simple model to predict a score based on a DNN activation.

    Args:
    - input_size (list): The size of the input tensor
    - hidden_dim (int): The size of the hidden layer
    - output_dim (int): The size of the output layer
    - dropout_prob (float): The probability of dropout

    Returns:
    - output (torch.Tensor): The output of the model
    """
    def __init__(
        self, 
        input_size: list, 
        hidden_dim: int = 128, 
        output_dim: int = 1, 
        dropout_prob=0
    ):
        super(ScorePredictor, self).__init__()

        out_channels = max(1, input_size[1] // 8)
        
        # 1x1 Conv Layer to reduce the number of input channels by a factor of 8
        self.conv1x1 = nn.Conv2d(
            in_channels=input_size[1], 
            out_channels=out_channels, 
            kernel_size=1
        )

        self.conv2x2_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2x2_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation, Norm, Dropout
        self.activation1 = nn.LeakyReLU()  # You can also use other activations like Swish or LeakyReLU
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Compute flattened size after convolution for the linear layers
        conv_output_size = out_channels * input_size[2] * input_size[3]
        # print(conv_output_size, out_channels, input_size[2], input_size[3])
        # Fully connected layer to reduce the output of the conv to hidden_dim
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.activation2 = nn.LeakyReLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final fully connected layer to output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # Pass through 1x1 convolution, activation, normalization, and dropout
        x = self.conv1x1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2x2_1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2x2_2(x)
        x = self.activation1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        # Flatten the output from the conv layer
        x = flatten(x, start_dim=1)
        # Fully connected layer to hidden_dim
        x = self.fc1(x)
        x = self.activation2(x)
        x = self.norm2(x)
        
        # Final output layer
        output = self.fc2(x)
        return output #sigmoid(output)



class ScorePredictionAdapter(nn.Module):
    def __init__(
        self,
        swivel: str,
        predictor: nn.Module,
        device: str = 'cuda:0',
        adversarial_training: bool = False,
        adversarial_prob: float = 0.9
    ):
        super().__init__()
        self.swivel = swivel
        self.predictor = predictor
        self.adversarial_training = adversarial_training
        self.adversarial_prob = adversarial_prob
        self.device = device
        self.active = True

        self.to(device)


    def on(self):
        self.active = True


    def off(self):
        self.active = False


    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        # this adapter only operates when turned on
        if self.active:
            self.score = self.predictor(x)

        else:
            pass
        return x



class ScorePredictionWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        adapters: nn.ModuleList,
        copy: bool = True,
    ):
        super().__init__()
        self.model           = deepcopy(model) if copy else model
        self.adapters        = adapters
        self.adapter_handles = {}
        self.transform       = False
        self.model.eval()
        self.hook_adapters()


    def hook_adapters(
        self,
    ) -> None:
        assert self.adapter_handles == {}, "Adapters already hooked"
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)


    def _get_hook(
        self,
        adapter: nn.Module
    ) -> Callable:
        def hook_fn(
            module: nn.Module, 
            x: Tuple[Tensor]
        ) -> Tensor:
            return adapter(x[0])
        
        return hook_fn
    

    def set_transform(self, transform: bool):
        self.transform = transform
        for adapter in self.adapters:
            adapter.transform = transform


    def turn_off_all_adapters(self):
        for adapter in self.adapters:
            adapter.off()

    
    def freeze_model(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def freeze_normalization_layers(self):
        for name, module in self.model.named_modules():
            if 'bn' in name:
                module.eval()


    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
    
        logits = self.model(x)
        
        self.output_per_adapter = [
            adapter.score for adapter in self.adapters if adapter.active
        ]
        if len(self.output_per_adapter) != 0:
            self.output_per_adapter = stack(self.output_per_adapter, dim=1)

        return logits



class ScorePredictionWrapperLightningModule(L.LightningModule):
    def __init__(
        self,
        wrapper: ScorePredictionWrapper,
        loss_fn: Callable,
        lr: float = 1e-6,
        num_classes: int = 4,
        patience: int = 10,
        adversarial_training: bool = False,
        adversarial_prob: float = 0.9,
        adversarial_step_size: float = 0.2,
        metadata: Dict[str, OmegaConf] = None
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                'wrapper', 
                'loss_fn', 
                'adversarial_batch',
                'training_step_outputs',
                'val_step_outputs',
                'predict_step_outputs',
                'eaurc'
            ]
        )
        self.wrapper = wrapper
        self.loss_fn = loss_fn
        self.lr = lr
        self.patience = patience
        self.adversarial_training = adversarial_training
        self.adversarial_prob = adversarial_prob
        self.adversarial_step_size = adversarial_step_size
        self.num_classes = num_classes
        self.metadata = metadata

        self.adversarial_batch = None
        self.prev_predicted_score = None
        self.prev_true_score = None
        self.non_adversarial_segmentation = None
        self.training_step_outputs, self.val_step_outputs, self.predict_step_outputs = [], [], []


    def forward(self, x):
        return self.wrapper(x)
    

    def adversarial_step(
        self,
        input: Tensor,
    ) -> Tensor:
        assert input.grad.data is not None, "Input tensor requires grad"
        grad = input.grad.data
        gradient_norm = grad.norm(2, dim=((1,2,3))).view(-1, 1, 1, 1) ** 2
        adv_input = input.data - self.gradient_factor * grad / (gradient_norm + 1e-8)
        self.zero_grad()

        return adv_input


    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        
        if self.adversarial_training and self.current_epoch > 1:
            mid_point = input.size(0)

            if self.adversarial_batch is None:
                input  = input.repeat(2, 1, 1, 1)
                target = target.repeat(2, 1, 1, 1)
            else:
                input  = cat([input,  self.adversarial_batch['input']])
                target = cat([target, self.adversarial_batch['target']])

            input = input.detach().requires_grad_(True)
            model_logits = self(input)
            if self.num_classes == 2:
                model_prediction = (model_logits > 0).long().detach()
            else:
                model_prediction = argmax(model_logits, dim=1, keepdim=True).detach()

            score_prediction = self.wrapper.output_per_adapter

            if self.prev_predicted_score is None:
                observed_delta  = 0.
                delta_step_size = 0.
                self.gradient_factor = 0.8

            else:
                observed_delta   = (self.prev_predicted_score - score_prediction[mid_point:].detach())
                delta_step_size  = observed_delta.mean() - self.adversarial_step_size
                self.gradient_factor -= 0.01 * delta_step_size

            adv_loss = score_prediction.sum()
            adv_loss.backward(retain_graph=True)

            # Take first half of the batch for adversarial training
            adv_input = self.adversarial_step(input)
            self.adversarial_batch = {
                'input': adv_input[:mid_point].detach(),
                'target': target[:mid_point].detach()
            }

            # due to lightning training logic, we do this last. Optimization step is handled elsewhere 
            loss, predicted_score, true_score = self.loss_fn(
                predicted_segmentation=model_prediction, 
                target_segmentation=target,
                prediction=score_prediction,
                num_classes=self.num_classes,
                return_scores=True
            )
            loss = loss.sum(1)
            
            if self.prev_true_score is None:
                true_score_delta = 0.
            else:
                true_score_delta = (self.prev_true_score - true_score[mid_point:].detach())
                      
            metrics = {
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score.detach().cpu(),
                'predicted_score': predicted_score.mean(1).detach().cpu(),
                'observed_delta': tensor(observed_delta).mean().view(-1, ).detach().cpu(),
                'true_score_delta': tensor(true_score_delta).mean().view(-1, ).detach().cpu(),
                'delta_step_size': tensor(delta_step_size).view(-1, ).detach().cpu(),
                'gradient_factor': tensor(self.gradient_factor).view(-1, ).detach().cpu(),
            }
            
            self.prev_predicted_score = score_prediction[:mid_point].detach()
            self.prev_true_score = true_score[:mid_point].detach()

        else:
            
            if self.current_epoch < 2:
                self.wrapper.adapters[0].adversarial_training = False
            else:
                self.wrapper.adapters[0].adversarial_training = True
            logits = self(input)

            if logits.shape[1] == 1:
                preds = (logits > 0).long().detach()
            else:
                preds = argmax(logits, dim=1, keepdim=True).detach()
            prediction = self.wrapper.output_per_adapter
            loss, predicted_score, true_score = self.loss_fn(
                predicted_segmentation=preds, 
                target_segmentation=target,
                prediction=prediction,
                num_classes=self.num_classes,
                return_scores=True
            )
            loss = loss.sum(1)
            
            metrics = {
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score.detach().cpu(),
                'predicted_score': predicted_score.detach().mean(1).cpu(),
            }

        self.training_step_outputs.append(metrics)

        return loss.mean()
    

    def on_train_epoch_end(self):
        outputs = {
            key: cat([d[key] for d in self.training_step_outputs], dim=0)
            for key in self.training_step_outputs[0].keys()
        }
        self.log('train_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True, logger=True)


        if self.adversarial_training and self.current_epoch > 1:
            self.log('observed_delta',  outputs['observed_delta'].mean(),  on_epoch=True, logger=True)
            self.log('delta_step_size', outputs['delta_step_size'].mean(), on_epoch=True, logger=True)
            self.log('true_score_delta', outputs['true_score_delta'].mean(), on_epoch=True, logger=True)
            self.log('gradient_factor', outputs['gradient_factor'].mean(), on_epoch=True, logger=True)
        else:
            self.log('unet_adv_vs_predictor_adv', 1, on_epoch=True, logger=True)

        self.training_step_outputs.clear()


    def shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        
        logits = self(input)
        if logits.shape[1] == 1:
            preds = (logits > 0).long().detach()
        else:
            preds = argmax(logits, dim=1, keepdim=True).detach()
        
        prediction = self.wrapper.output_per_adapter

        loss, predicted_score, true_score = self.loss_fn(
            predicted_segmentation=preds, 
            target_segmentation=target,
            prediction=prediction,
            num_classes=self.num_classes,
            return_scores=True
        )

        metrics =  {
            'loss': loss.sum(1).detach().cpu(),
            'true_score': true_score.detach().cpu(),
            'predicted_score': predicted_score.detach().cpu()       
        }

        return metrics



    def validation_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        self.val_step_outputs.append(metrics)
        return metrics



    def on_validation_epoch_end(self):
        outputs = {
            key: cat([d[key] for d in self.val_step_outputs], dim=0) 
            for key in self.val_step_outputs[0].keys()
        }
        mae = (outputs['predicted_score'] - outputs['true_score']).abs().mean(0)
        if self.current_epoch > 1:
            self.log('val_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('val_loss', 1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_min_mae', mae.mean().item(), on_epoch=True, logger=True)
        for i, m in enumerate(mae):
            self.log(f'val_mae_{i}', m, on_epoch=True, logger=True)

        risk = 1 - outputs['true_score'].squeeze(1)
        predicted_risks = 1 - outputs['predicted_score']

        # iterate over the predictors with shape [B, n_predictors]
        for i, predicted_risk in enumerate(predicted_risks.T):
            eaurc = eAURC(predicted_risk, risk, ret_curves=False)
            self.log(f'eaurc_{i}', eaurc, on_epoch=True, logger=True)
        
        self.val_step_outputs.clear()


    def predict_step(self, batch, batch_idx, dataloader_idx):
        metrics = self.shared_eval_step(batch, batch_idx, dataloader_idx)
        self.predict_step_outputs.append(metrics)

        return metrics
    

    def configure_optimizers(self):
        optimizer = Adam(self.wrapper.adapters.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }



class ScorePredictionFinetunerLightningModule(L.LightningModule):
    def __init__(
        self,
        wrapper: ScorePredictionWrapper,
        oracle: nn.Module,
        loss_fn: Callable,
        lr: float = 1e-6,
        num_classes: int = 4,
        patience: int = 10,
        adversarial_training: bool = False,
        adversarial_prob: float = 0.9,
        adversarial_step_size: float = 0.2,
        metadata: Dict[str, OmegaConf] = None
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                'wrapper', 
                'oracle', 
                'loss_fn', 
                'adversarial_batch',
                'training_step_outputs',
                'val_step_outputs',
                'predict_step_outputs',
                'eaurc'
            ]
        )
        self.wrapper = wrapper
        self.oracle = oracle
        self.loss_fn = loss_fn
        self.lr = lr
        self.patience = patience
        self.adversarial_training = adversarial_training
        self.adversarial_prob = adversarial_prob
        self.adversarial_step_size = adversarial_step_size
        self.num_classes = num_classes
        self.metadata = metadata

        self.adversarial_batch = None
        self.prev_predicted_score = None
        self.training_step_outputs, self.val_step_outputs, self.predict_step_outputs = [], [], []


    def forward(self, x):
        return self.wrapper(x)
    

    def adversarial_step(
        self,
        input: Tensor,
    ) -> Tensor:
        assert input.grad.data is not None, "Input tensor requires grad"
        grad = input.grad.data
        gradient_norm = grad.norm(2, dim=((1,2,3))).view(-1, 1, 1, 1) ** 2
        adv_input = input.data - self.gradient_factor * grad / (gradient_norm + 1e-8)
        self.zero_grad()

        return adv_input


    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        
        if self.adversarial_training:
            # new code
            mid_point = input.size(0)

            if self.adversarial_batch is None:
                input  = input.repeat(2, 1, 1, 1)
                target = target.repeat(2, 1, 1, 1)
            else:
                input  = cat([input,  self.adversarial_batch['input']])
                target = cat([target, self.adversarial_batch['target']])

            input = input.detach().requires_grad_(True)
            
            # finetuning logic changed this
            _ = self(input)
            model_logits = self.oracle(input).detach()
            #
            if self.num_classes == 2:
                model_prediction = (model_logits > 0).long().detach()
            else:
                model_prediction = argmax(model_logits, dim=1, keepdim=True).detach()

            score_prediction = self.wrapper.output_per_adapter

            if self.prev_predicted_score is None:
                observed_delta  = 0.
                delta_step_size = 0.
                self.gradient_factor = 1.

            else:
                observed_delta   = (self.prev_predicted_score - score_prediction[mid_point:].detach())
                delta_step_size  = observed_delta.mean() - self.adversarial_step_size
                self.gradient_factor -= 0.01 * delta_step_size

            adv_loss = score_prediction.sum()
            adv_loss.backward(retain_graph=True)

            # Take first half of the batch for adversarial training
            adv_input = self.adversarial_step(input)
            self.adversarial_batch = {
                'input': adv_input[:mid_point].detach(),
                'target': target[:mid_point].detach()
            }

            # due to lightning training logic, we do this last. Optimization step is handled elsewhere 
            loss, predicted_score, true_score = self.loss_fn(
                predicted_segmentation=model_prediction, 
                target_segmentation=target,
                prediction=score_prediction,
                num_classes=self.num_classes,
                return_scores=True
            )
            loss = loss.sum(1)
            
            metrics = {
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score.detach().cpu(),
                'predicted_score': predicted_score.mean(1).detach().cpu(),
                'observed_delta': tensor(observed_delta).mean().view(-1, ).detach().cpu(),
                'delta_step_size': tensor(delta_step_size).view(-1, ).detach().cpu(),
                'gradient_factor': tensor(self.gradient_factor).view(-1, ).detach().cpu(),
            }
            
            self.prev_predicted_score = score_prediction[:mid_point].detach()

        else:
            
            # finetuning logic changed this
            _ = self(input)
            logits = self.oracle(input).detach()
            #
            if logits.shape[1] == 1:
                preds = (logits > 0).long().detach()
            else:
                preds = argmax(logits, dim=1, keepdim=True).detach()
            prediction = self.wrapper.output_per_adapter
            loss, predicted_score, true_score = self.loss_fn(
                predicted_segmentation=preds, 
                target_segmentation=target,
                prediction=prediction,
                num_classes=self.num_classes,
                return_scores=True
            )
            loss = loss.sum(1)
            
            metrics = {
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score.detach().cpu(),
                'predicted_score': predicted_score.detach().mean(1).cpu(),
            }

        self.training_step_outputs.append(metrics)

        return loss.mean()
    

    def on_train_epoch_end(self):
        outputs = {
            key: cat([d[key] for d in self.training_step_outputs], dim=0)
            for key in self.training_step_outputs[0].keys()
        }
        self.log('train_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True, logger=True)


        if self.adversarial_training:
            self.log('observed_delta',  outputs['observed_delta'].mean(),  on_epoch=True, logger=True)
            self.log('delta_step_size', outputs['delta_step_size'].mean(), on_epoch=True, logger=True)
            self.log('gradient_factor', outputs['gradient_factor'].mean(), on_epoch=True, logger=True)
        else:
            self.log('unet_adv_vs_predictor_adv', 1, on_epoch=True, logger=True)

        self.training_step_outputs.clear()


    def shared_eval_step(self, batch, batch_idx, dataloader_idx=None):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        
        # finetuning logic changed this
        _ = self(input)
        logits = self.oracle(input).detach()
        #
        if logits.shape[1] == 1:
            preds = (logits > 0).long().detach()
        else:
            preds = argmax(logits, dim=1, keepdim=True).detach()
        
        prediction = self.wrapper.output_per_adapter

        loss, predicted_score, true_score = self.loss_fn(
            predicted_segmentation=preds, 
            target_segmentation=target,
            prediction=prediction,
            num_classes=self.num_classes,
            return_scores=True
        )

        metrics =  {
            'loss': loss.sum(1).detach().cpu(),
            'true_score': true_score.detach().cpu(),
            'predicted_score': predicted_score.detach().cpu()       
        }

        return metrics



    def validation_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        self.val_step_outputs.append(metrics)
        return metrics



    def on_validation_epoch_end(self):
        outputs = {
            key: cat([d[key] for d in self.val_step_outputs], dim=0) 
            for key in self.val_step_outputs[0].keys()
        }
        mae = (outputs['predicted_score'] - outputs['true_score']).abs().mean(0)

        self.log('val_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_min_mae', mae.mean().item(), on_epoch=True, logger=True)
        for i, m in enumerate(mae):
            self.log(f'val_mae_{i}', m, on_epoch=True, logger=True)

        risk = 1 - outputs['true_score'].squeeze(1)
        predicted_risks = 1 - outputs['predicted_score']

        # iterate over the predictors with shape [B, n_predictors]
        for i, predicted_risk in enumerate(predicted_risks.T):
            eaurc = eAURC(predicted_risk, risk, ret_curves=False)
            self.log(f'eaurc_{i}', eaurc, on_epoch=True, logger=True)
        
        self.val_step_outputs.clear()


    def predict_step(self, batch, batch_idx, dataloader_idx):
        metrics = self.shared_eval_step(batch, batch_idx, dataloader_idx)
        self.predict_step_outputs.append(metrics)

        return metrics
    

    def configure_optimizers(self):
        optimizer = Adam(self.wrapper.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }
    

