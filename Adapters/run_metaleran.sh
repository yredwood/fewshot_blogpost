gpu=6
N=10 # n-way 
K=5 # k-shot
M=1
Q=15
A=1
F=0
U=0 # 1 for new adapter, 0 for existing one
config="cl_miniImagenet"  #for learn
name=AdaptNet_${config}_aug${A}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

#echo $log
#exec &> >(tee -a "$log")

# ------------------------------------------------------------------------
# first you need to pretrain the network
#CUDA_VISIBLE_DEVICES=$gpu python -u learning.py --maxe 100 \
#    --config $config --bs 128 --gpuf 0.91 --name $name --aug=$A --na 2 \
#    --pr 1 --train 0 --vali 600  # test code


cconfig="miniImagenet"
arch="res"
function ftest(){
    CUDA_VISIBLE_DEVICES=7 python metalearning.py --gpuf 0.91 \
        --config $cconfig --qs 15 --nw 5 --ks 1 \
        --mbsize 1 --aug 0 --fix_univ 0 --use_adapt 0 \
        --tr 0 --vali 600 --name $1 --pr none --arch $arch
}

function ftrain(){
    CUDA_VISIBLE_DEVICES=$1 python metalearning.py --gpuf 0.91 --lr 0.001 \
        --config $cconfig --qs 50 --mbsize 1 --aug 0 --fix_univ 0 --arch $arch \
        --use_adapt 0 --maxe $2 --nw $3 --ks $4 --name $5 --pr $6 
}


# ------------------------------------------------------------------------

N=5
K=1
P="../models/AdaptNet_cl_miniImagenet_aug1/cl_miniImagenet.ckpt"
P="../models/AdaptNet_cl_miniImagenet_aug1_res/cl_miniImagenet.ckpt"
#P=none
if [ "$P" = "none" ]; then
    postfix=None
else
    postfix=Pretrained
fi 
name=Cur_${cconfig}_${arch}_${M}${N}${K}_${postfix}

# step 1 
#ftrain 5 50 $N $K $name $P

#N=5
#K=1
#P="../models/Cur_miniImagenet_res_121_Pretrained"
#name=Cur_${cconfig}_${arch}_${M}${N}${K}_${postfix}
#ftrain 4 100 $N $K $name $P
ftest $name







## step 2
#P=../models/${name}/miniImagenet.ckpt
#N=10
#K=1
#name=Cur_${cconfig}_${M}${N}${K}
#ftrain 1 30 $N $K $name $P
#ftest $name
#
#
## step 3
#P=../models/${name}/miniImagenet.ckpt
#N=5
#K=1
#name=Cur_${cconfig}_${M}${N}${K}
#ftrain 1 30 $N $K $name $P
#ftest $name

#CUDA_VISIBLE_DEVICES=$gpu python -u metalearning.py --gpuf 0.91 --pr $P \
#    --config $config --qs $Q --nw $N --ks $K --name $name --maxe 2





# ------------------------------------------------------------------------
A=0
P=1
config="miniImagenet" # for metalearn
name=AdaptNet_${config}_aug${A}_fix${F}_U${U}_${M}${N}${K}
#CUDA_VISIBLE_DEVICES=$gpu python -u metalearning.py --gpuf 0.91 --pr $P \
#    --config $config --qs $Q --nw $N --ks $K --name $name \
#    --mbsize $M --aug $A --fix_univ $F --maxe 20 --use_adapt $U \

