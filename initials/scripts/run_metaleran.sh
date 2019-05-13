
N=5 # n-way
K=1 # k-shot
M=1
Q=15

cconfig="miniImagenet"
arch="simple"
meta_arch="proto"
function ftest(){
    CUDA_VISIBLE_DEVICES=6 python metalearning.py --gpuf 0.41 \
        --config $cconfig --qs 15 --nw 5 --ks 1 --mbsize 1 \
        --tr 0 --vali 600 --name $1 --pr none --arch $arch --meta_arch $meta_arch \
        --cl tsklist_random_large5w.npy
}

function ftrain(){
    CUDA_VISIBLE_DEVICES=$1 python metalearning.py --gpuf 0.41 --vali 100 \
        --config $cconfig --qs $Q --mbsize $M --aug 0 --arch $arch --meta_arch $meta_arch \
        --maxe $2 --nw $3 --ks $4 --name $5 --lr $6 --epl $7 --pr $8 \
        --cl tsklist_random_large5w.npy
}


# ------------------------------------------------------------------------

#LR=2e-3
#EL=0.7

#LR=2e-3
#EL=0.5,0.8

#LR=1e-3
#EL=0.9

#1
LR=1e-3
EL=0.7

#2
#LR=1e-2
#EL=0.5,0.8

#3
#LR=5e-3
#EL=0.7


#P="../models/AdaptNet_cl_miniImagenet_aug1/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_simple_maml/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_res/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_res12/cl_miniImagenet.ckpt"
#P="../models/Metalearn_miniImagenet_simple_151_Pretrained/miniImagenet.ckpt"
P="../models/Learner_cl_miniImagenet_simple_proto/cl_miniImagenet.ckpt"
#P=none
if [ "$P" = "none" ]; then
    postfix=None
else
    postfix=Pretrained
fi 
name=Metalearn_${cconfig}_${arch}_${M}${N}${K}_${postfix}_${LR}


# step 1 
#ftrain 0 100 $N $K $name $LR $EL $P 
ftest $name

#N=5
#K=1
#P="../models/Cur_miniImagenet_res_121_Pretrained"
#name=Cur_${cconfig}_${arch}_${M}${N}${K}_${postfix}
#ftrain 4 100 $N $K $name $P
#ftest $name







