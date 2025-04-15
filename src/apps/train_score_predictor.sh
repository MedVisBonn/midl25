# Behavior
CUDA_DEVICE=5
TRAIN=True
TEST=True
LOGGING=False

# Training
NON_EMPTY_TARGET=True
N_BATCHES=100
N_EPOCHS=100
MODEL_POSTFIX='beta01'
ADVERSARIAL_PROB=0.5
ADVERSARIAL_STEP_SIZE=0.1
LOSS_FN='surface'

ADVERSARIAL_TRAINING=True
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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_score_predictor.py \
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
    +trainer=score_predictor_trainer \
    ++trainer.limit_train_batches="$N_BATCHES" \
    ++trainer.max_epochs="$N_EPOCHS" \
    ++trainer.logging="$LOGGING" \
    ++trainer.ckpt_name="$CKPT_NAME"