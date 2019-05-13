
N=20 # n-way
K=500 # k-shot
M=1
Q=100

cconfig="tieredImagenet"
arch="wdres"
meta_arch="proto"
#cluster_npy="cconfig20.npy"
cluster_npy="tsklist_random_tiered100w.npy"
function ftest(){
    CUDA_VISIBLE_DEVICES=6 python metalearning.py --gpuf 0.91 \
        --config $cconfig --qs 15 --nw 5 --ks 1 --mbsize 1 \
        --tr 0 --vali 600 --name $1 --pr none --arch $arch --meta_arch $meta_arch \
        --batch_size 128
}

function ftrain(){
    CUDA_VISIBLE_DEVICES=$1 python metalearning.py --gpuf 0.91 --vali 100 \
        --config $cconfig --qs $Q --mbsize $M --aug 0 --arch $arch --meta_arch $meta_arch \
        --maxe $2 --nw $3 --ks $4 --name $5 --lr $6 --epl $7 --pr $8 --pn $9 \
        --cl clusters/$cluster_npy \
        --batch_size 128
}


# ------------------------------------------------------------------------

#LR=2e-3
#EL=0.7

#LR=2e-3
#EL=0.5,0.8

#LR=1e-3
#EL=0.9

#1
#LR=2e-2
#EL=0.7

#2
LR=1e-3
EL=0.5,0.8

#3
#LR=5e-3
#EL=0.7


#P="../models/AdaptNet_cl_miniImagenet_aug1/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_simple_maml/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_res12/cl_miniImagenet.ckpt"
#P="../models/Metalearn_miniImagenet_simple_151_Pretrained/miniImagenet.ckpt"
P="../models/Learner_cl_miniImagenet_simple_proto/cl_miniImagenet.ckpt"
#P="../models/Learner_cl_miniImagenet_res/cl_miniImagenet.ckpt"
#P=none
if [ "$P" = "none" ]; then
    postfix=None
else
    postfix=Pretrained
fi 
name=Metalearn_${cconfig}_${arch}_${M}${N}${K}_${postfix}_${cluster_npy}


pn=2
gpu=$(( $pn - 0 ))
#gpu=$pn
# step 1 
ftrain $gpu 100 $N $K $name $LR $EL $P $pn
#ftest $name

#N=5
#K=1
#P="../models/Cur_miniImagenet_res_121_Pretrained"
#name=Cur_${cconfig}_${arch}_${M}${N}${K}_${postfix}
#ftrain 4 100 $N $K $name $P
#ftest $name







