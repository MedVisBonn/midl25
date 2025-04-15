import sys
from omegaconf import OmegaConf
import hydra
sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module, get_unet_module_trainer



@hydra.main(
    config_path='../configs', 
    version_base=None
)
def main(cfg):

    # init datamodule
    datamodule = get_data_module(
        cfg=cfg.data
    )

    # init model
    model = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg),
        load_from_checkpoint=False
    )

    # init trainer
    trainer = get_unet_module_trainer(
        data_cfg=cfg.data,
        model_cfg=cfg.unet,
        trainer_cfg=cfg.trainer
    )

    # train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()