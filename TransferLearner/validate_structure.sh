gpu=5
W=10 # n-way 
S=150 # k-shot
config="cl_cifar10"
name=transferNet_scratch
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

#echo $log
#exec &> >(tee -a "$log")


# first you need to pretrain the network
CUDA_VISIBLE_DEVICES=$gpu python -u pretrain.py --lr 1e-3 \
    --config $config \
    --pr 1 --train 0 --vali 600  # test code


# then test transfer learning. if --pr 0 then 
# it learns from scratch
#CUDA_VISIBLE_DEVICES=$gpu python -u transfer.py \
#    --config $config --nw $W --ks $S --vali 600 --pr 1 \

