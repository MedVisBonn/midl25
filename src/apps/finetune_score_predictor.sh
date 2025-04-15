# Behavior
CUDA_DEVICE=7
TRAIN=True
TEST=True
LOGGING=True

# Training
NON_EMPTY_TARGET=True
N_BATCHES=100
N_EPOCHS=25
LR=0.00000316
MODEL_POSTFIX=None
ADVERSARIAL_PROB=0.5

ADVERSARIAL_STEP_SIZE=0.1
ADVERSARIAL_TRAINING=True
LOSS_FN='dice'
CKPT_NAME='None'
DATA='mnmv2'


if [ "$DATA" = 'mnmv2' ]; then
    UNET_OUT_DIM=4
    PRED_OUT_DIM=4
    NUM_CLASSES=4
    DOMAIN="Symphony"
elif [ "$DATA" = 'pmri' ]; then
    UNET_OUT_DIM=1
    PRED_OUT_DIM=2
    NUM_CLASSES=2
    DOMAIN="RUNMC"

fi

if [ "$ADVERSARIAL_TRAINING" = "True" ]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=64
fi

CKPTS="mnmv2_Symphony_dice_adversarial_2025-01-15-11-29.ckpt
mnmv2_Symphony_dice_adversarial_2025-01-25-20-32.ckpt
mnmv2_Symphony_dice_adversarial_2025-01-25-21-17.ckpt
mnmv2_Symphony_dice_adversarial_2025-01-25-22-00.ckpt
mnmv2_Symphony_dice_adversarial_2025-01-25-22-42.ckpt"

while IFS= read -r WARPPER_CKPT_NAME; do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python finetune_score_predictor.py \
        +train="$TRAIN" \
        +test="$TEST" \
        +data="$DATA" \
        ++data.domain="$DOMAIN" \
        ++data.non_empty_target="$NON_EMPTY_TARGET" \
        +unet=monai_unet \
        ++unet.out_channels="$UNET_OUT_DIM" \
        +model=score_predictor \
        ++model.adapter_output_dim="$PRED_OUT_DIM" \
        ++model.num_classes="$NUM_CLASSES" \
        ++model.name="$MODEL_POSTFIX" \
        ++model.adversarial_training="$ADVERSARIAL_TRAINING" \
        ++model.adversarial_prob="$ADVERSARIAL_PROB" \
        ++model.adversarial_step_size="$ADVERSARIAL_STEP_SIZE" \
        ++model.loss_fn="$LOSS_FN" \
        ++model.wrapper_ckpt_name="$WARPPER_CKPT_NAME" \
        ++model.lr="$LR" \
        +trainer=score_predictor_trainer \
        ++trainer.finetune=True \
        ++trainer.limit_train_batches="$N_BATCHES" \
        ++trainer.max_epochs="$N_EPOCHS" \
        ++trainer.logging="$LOGGING" \
        ++trainer.ckpt_name="$CKPT_NAME"
done <<< "$CKPTS"





# if [ "$DATA"]
# WARPPER_CKPT_NAME='mnmv2_Symphony_dice_adversarial_2025-01-15-11-29.ckpt'
# WARPPER_CKPT_NAME='mnmv2_Symphony_dice_normal_2025-01-15-11-27.ckpt'
# WARPPER_CKPT_NAME='pmri_RUNMC_dice_adversarial_2025-01-15-11-30.ckpt'
# WARPPER_CKPT_NAME='pmri_RUNMC_dice_normal_2025-01-15-11-30.ckpt'

    # if [ "$ADVERSARIAL_TRAINING" = "True" ]; then
    #     if [ "$LOSS_FN" = 'dice' ]; then
    #         WARPPER_CKPT_NAME='mnmv2_Symphony_dice_adversarial_2025-01-15-11-29.ckpt'
    #     elif [ "$LOSS_FN" = 'surface' ]; then
    #         WARPPER_CKPT_NAME='mnmv2_Symphony_surface_adversarial_2025-01-17-00-25.ckpt' #TODO
    #     fi
    # elif [ "$ADVERSARIAL_TRAINING" = "False" ]; then
    #     if [ "$LOSS_FN" = 'dice' ]; then
    #         WARPPER_CKPT_NAME='mnmv2_Symphony_dice_normal_2025-01-15-11-27.ckpt'
    #     elif [ "$LOSS_FN" = 'surface' ]; then
    #         WARPPER_CKPT_NAME='mnmv2_Symphony_surface_normal_2025-01-17-00-24.ckpt' #TODO
    #     # if [ "$ADVERSARIAL_TRAINING" = "True" ]; then
    #     if [ "$LOSS_FN" = 'dice' ]; then
    #         WARPPER_CKPT_NAME='pmri_RUNMC_dice_adversarial_2025-01-15-11-30.ckpt'
    #     elif [ "$LOSS_FN" = 'surface' ]; then
    #         WARPPER_CKPT_NAME='pmri_RUNMC_surface_adversarial_2025-01-17-00-28.ckpt' #TODO
    #     fi
    # elif [ "$ADVERSARIAL_TRAINING" = "False" ]; then
    #     if [ "$LOSS_FN" = 'dice' ]; then
    #         WARPPER_CKPT_NAME='pmri_RUNMC_dice_normal_2025-01-15-11-30.ckpt'
    #     elif [ "$LOSS_FN" = 'surface' ]; then
    #         WARPPER_CKPT_NAME='pmri_RUNMC_surface_normal_2025-01-17-00-37.ckpt' #TODO
    #     fi
    # fi    fi

# for DATA in 'mnmv2' 'pmri'; do

# for DATA in 'mnmv2' 'pmri'; do