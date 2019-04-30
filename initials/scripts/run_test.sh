N=5
K=500
Q=100

#P="../models/Learner_cl_miniImagenet_simple/cl_miniImagenet.ckpt"
#P="../models/Metalearn_miniImagenet_simple_151_None/miniImagenet.ckpt"
P="../models/Metalearn_miniImagenet_simple_151_Pretrained_1e-3/miniImagenet.ckpt"
#P="none"
C="miniImagenet"
Ar="simple"

CUDA_VISIBLE_DEVICES=5 python test.py \
    --qs $Q --nw $N --ks $K \
    --pr $P --gpuf 0.94 --config $C \
    --arch $Ar --train 0 --vali 120 \


#
