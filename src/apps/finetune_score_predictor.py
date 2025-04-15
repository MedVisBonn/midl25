import sys, os
import pickle as pkl
from omegaconf import OmegaConf
import hydra
from torch import nn
sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from model.score_adapter import (
    get_score_prediction_module, 
    get_score_prediction_finetune_module,
    get_score_prediction_module_trainer,
    clean_predictions,
    collect_eval_from_predictions,
    get_bar_plots_and_table,
    get_risk_coverage_curves
)

# UNET_CKPTS = {
#     "mnmv2": {
#         'dropout-0-0': 'mnmv2_symphony_dropout-0-0_2025-01-14-15-20',
#         'dropout-0-1': 'mnmv2_symphony_dropout-0-1_2025-01-14-15-19',
#         'dropout-0-2': 'mnmv2_symphony_dropout-0-2_2025-01-14-15-18',
#         'dropout-0-3': 'mnmv2_symphony_dropout-0-5_2025-01-14-15-20'
#     },
#     'pmri': {
#         'dropout-0-0': 'pmri_runmc_dropout-0-0_2025-01-14-15-58',
#         'dropout-0-1': 'pmri_runmc_dropout-0-1_2025-01-14-15-58',
#         'dropout-0-2': 'pmri_runmc_dropout-0-2_2025-01-14-15-58',
#         'dropout-0-5': 'pmri_runmc_dropout-0-5_2025-01-14-15-58'
#     },
# }

UNET_CKPTS = {
    "mnmv2": 'mnmv2_symphony_dropout-0-1_2025-01-14-15-19',
    'pmri': 'pmri_runmc_dropout-0-1_2025-01-14-15-58',
}


# HAUSDORFF_SIGMAS = {
#     "mnmv2": {
#         'siemens': 1.6761,
#         'philips': 1.2080, 
#         'ge': 1.6435
#     }, 
#     'pmri': {
#         'siemens': 2.2983,
#         'philips': 2.1584, 
#         'ge': 2.6745
#     }
# }


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
    ckpt = UNET_CKPTS[cfg.data.dataset] #[cfg.data.domain]
    cfg.unet.checkpoint_path = f'{cfg.trainer.root_dir}{cfg.unet.checkpoint_dir}{ckpt}.ckpt'


    cfg.unet.dropout = 0.0
    oracle = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg.unet),
        load_from_checkpoint=True
    ).model

    unet = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg.unet),
        load_from_checkpoint=True
    ).model
    
    # cfg.model.hausdorff_sigma = HAUSDORFF_SIGMAS[cfg.data.dataset][cfg.data.domain]
    wrapper_ckpt = cfg.trainer.root_dir + cfg.trainer.ckpt_dir + cfg.model.wrapper_ckpt_name
    wrapper = get_score_prediction_module(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        unet=unet,
        metadata=OmegaConf.to_container(cfg.model), #TODO
        ckpt=wrapper_ckpt
    ).wrapper

    print(OmegaConf.to_yaml(cfg))


    if cfg.train == True:
        ckpt = None
    else:
        ckpt = cfg.trainer.root_dir + cfg.trainer.ckpt_dir + cfg.trainer.ckpt_name
        assert os.path.exists(ckpt), f'Checkpoint not found at {ckpt}'

    model = get_score_prediction_finetune_module(
        wrapper=wrapper,
        oracle=oracle,
        model_cfg=cfg.model,
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

        # make graphs
        sources = cfg.model.name
        fig_scores, wide_df = get_bar_plots_and_table(
            [evaluation], 
            [sources], 
            scores=['corr', 'mae', 'eaurc']
        )
        fig_curves = get_risk_coverage_curves([evaluation], sources)

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
        wide_df.to_csv(f'{save_dir}/scores.csv')
        fig_scores.savefig(f'{save_dir}/barplots.png')
        fig_curves.savefig(f'{save_dir}/risk_coverage_curves.png')



if __name__ == '__main__':
    main()