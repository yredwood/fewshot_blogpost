
M=1
Q=15

cconfig="miniImagenet"
arch="simple"
function ftest(){
    CUDA_VISIBLE_DEVICES=7 python metalearning.py --gpuf 0.91 \
        --config $cconfig --qs 15 --nw 5 --ks 1 --mbsize 1 \
        --tr 0 --vali 600 --name $1 --pr none --arch $arch
}

function ftrain(){
    CUDA_VISIBLE_DEVICES=$1 python metalearning.py --gpuf 0.91 \
        --config $cconfig --qs $Q --mbsize $M --aug 0 --arch $arch \
        --maxe $2 --nw $3 --ks $4 --name $5 --lr $6 --epl $7 --pr $8
}


gpu=5
# ------------------------------------------------------------------------
# path1
LR=1e-2
EL=0.9

N=5 # n-way
K=30 # k-shot

P="../models/Learner_cl_miniImagenet_simple/cl_miniImagenet.ckpt"
#P=none
if [ "$P" = "none" ]; then
    postfix=None
else
    postfix=Pretrained
fi 
name=Metalearn_${cconfig}_${arch}_${M}${N}${K}_${postfix}_${LR}

# step 1 
#ftrain $gpu 20 $N $K $name $LR $EL $P 
#ftest $name

#---------------------------------------------------------
# path2 
P=../models/${name}/${cconfig}.ckpt
LR=1e-2
EL=0.9

N=5
K=15
name=Metalearn_${cconfig}_${arch}_${M}${N}${K}_${postfix}_${LR}
#ftrain $gpu 20 $N $K $name $LR $EL $P 
#ftest $name
#---------------------------------------------------------
# path3
P=../models/${name}/${cconfig}.ckpt
LR=1e-2
EL=0.5,0.8

N=5
K=1
name=Metalearn_${cconfig}_${arch}_${M}${N}${K}_${postfix}_${LR}c
ftrain $gpu 40 $N $K $name $LR $EL $P 
ftest $name







