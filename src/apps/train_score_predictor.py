import sys, os
import pickle as pkl
from omegaconf import OmegaConf
import hydra
sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from model.score_adapter import (
    get_score_prediction_module, 
    get_score_prediction_module_trainer
)
from utils import (
    clean_predictions,
    collect_eval_from_predictions,
)

UNET_CKPTS = {
    "mnmv2": 'mnmv2_symphony_dropout-0-1_2025-01-14-15-19',
    'pmri': 'pmri_runmc_dropout-0-1_2025-01-14-15-58',
}


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
    ckpt = UNET_CKPTS[cfg.data.dataset]
    cfg.unet.checkpoint_path = f'{cfg.trainer.root_dir}{cfg.unet.checkpoint_dir}{ckpt}.ckpt'
    unet = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg),
        load_from_checkpoint=True
    ).model

    if cfg.train == True:
        ckpt = None
    else:
        ckpt = cfg.trainer.root_dir + cfg.trainer.ckpt_dir + cfg.trainer.ckpt_name
        assert os.path.exists(ckpt), f'Checkpoint not found at {ckpt}'

    print(OmegaConf.to_yaml(cfg))
    
    model = get_score_prediction_module(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        unet=unet,
        metadata=OmegaConf.to_container(cfg.model), #TODO
        ckpt=ckpt
    )

    # init trainer
    trainer = get_score_prediction_module_trainer(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        trainer_cfg=cfg.trainer
    )
    
    # train
    if cfg.train == True:
        trainer.fit(model=model, datamodule=datamodule)
        ckpt = trainer.checkpoint_callback.best_model_path

    # test
    if cfg.test == True:
        prediction = trainer.predict(
            model=model, 
            datamodule=datamodule,
            ckpt_path=ckpt
        )

        # clean / format predictions and calc metrics
        prediction_clean = clean_predictions(prediction, datamodule, model)
        evaluation = collect_eval_from_predictions(prediction_clean)

        # save results
        model_name = ckpt.split('/')[-1].replace('0.', '0-').split('.')[0]
        save_dir = f'{cfg.trainer.root_dir}/results/{model_name}'
        os.makedirs(save_dir, exist_ok=True)
        evaluation = {
            'metadata': {
                'data': OmegaConf.to_container(cfg.data),
                'unet': OmegaConf.to_container(cfg.unet),
                'model': OmegaConf.to_container(cfg.model),
                'trainer': OmegaConf.to_container(cfg.trainer)
            },
            'evaluation': evaluation
        }
        with open(f'{save_dir}/eval_data.pkl', 'wb') as f:
            pkl.dump(evaluation, f)



if __name__ == '__main__':
    main()