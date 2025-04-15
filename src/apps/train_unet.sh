CUDA_DEVICE=7

DROPOUT=0.5
#TODO: add dropout and augmentation control
# 'mnmv2'
DATA='pmri'
if [ "$DATA" = 'mnmv2' ]; then
    OUT_DIM=4
    DOMAIN="Symphony"
elif [ "$DATA" = 'pmri' ]; then
    OUT_DIM=1
    DOMAIN="RUNMC"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_unet.py \
    +data="$DATA" \
    ++data.domain="$DOMAIN" \
    +unet=monai_unet \
    ++unet.out_channels="$OUT_DIM" \
    ++unet.dropout="$DROPOUT" \
    +trainer=unet_trainer