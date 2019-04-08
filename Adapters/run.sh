gpu=6
N=5 # n-way 
K=1 # k-shot
config="miniimg64"
arch="resadapt"
name=AdaptNet_${config}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

#echo $log
#exec &> >(tee -a "$log")

# first you need to pretrain the network
#CUDA_VISIBLE_DEVICES=$gpu python -u learning.py --lr 1e-1 \
#    --config $config --bs 128 --gpuf 0.91 --name $name \
#    --pr 1 --train 0 --vali 600  # test code


CUDA_VISIBLE_DEVICES=$gpu python -u metalearning.py --lr 1e-1 --gpuf 0.91 \
    --config $config --qs 15 --nw $N --ks $K --name $name --mbsize 1 \
    --train 0 --vali 600 --pr 1 


# then test transfer learning. if --pr 0 then 
# it learns from scratch
#CUDA_VISIBLE_DEVICES=$gpu python -u transfer.py \
#    --config $config --nw $W --ks $S --vali 600 --pr 1 \

