gpu=3
N=10 # n-way 
K=5 # k-shot
M=1
Q=15
A=1
F=0
U=0 # 1 for new adapter, 0 for existing one
arch="simple"
meta_arch="maml"
config="cl_miniImagenet"  #for learn
#config="cl_miniImagenet"
name=Learner_${config}_${arch}_${meta_arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

#echo $log
#exec &> >(tee -a "$log")

# ------------------------------------------------------------------------
# first you need to pretrain the network
CUDA_VISIBLE_DEVICES=$gpu python -u learning.py --maxe 200 --arch $arch \
    --config $config --bs 128 --gpuf 0.91 --name $name --aug $A --meta_arch $meta_arch \
#    --pr 1 --train 0 --vali 600  # test code


## ------------------------------------------------------------------------
#A=0
#P=1
#config="miniImagenet" # for metalearn
#name=AdaptNet_${config}_aug${A}_fix${F}_U${U}_${M}${N}${K}
##CUDA_VISIBLE_DEVICES=$gpu python -u metalearning.py --gpuf 0.91 --pr $P \
##    --config $config --qs $Q --nw $N --ks $K --name $name \
##    --mbsize $M --aug $A --fix_univ $F --maxe 20 --use_adapt $U \
#
