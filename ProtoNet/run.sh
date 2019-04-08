######  GUIDE  ######
# K : training shots 
# tk : test shots

# logging makes pdb broken
#echo $log
#exec &> >(tee -a "$log")

Q=15
MB=4
N=5
K=1
data="miniImagenet"

gpu=3
arch="simple"
name=${N}w${K}s${Q}q${MB}m_protonet_${data}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}
#
## to get test result, uncomment last line
#CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
#    --nw $N --ks $K --name $name --dset ${data} \
#    --arch $arch --gpuf 0.93 --qs $Q --mbsize $MB \
#    --pr 1 --train 0 --vali 600 --tk 0
#echo $name

# results of 5way 1shot (simplenet)
# 

# ----------------------------------------------------------

gpu=4
arch="vggnet"
name=${N}w${K}s${Q}q${MB}m_protonet_${data}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

# to get test result, uncomment last line
#CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
#    --nw $N --ks $K --name $name --dset ${data} \
#    --arch $arch --gpuf 0.93 --qs $Q --mbsize $MB \
#    --pr 1 --train 0 --vali 600 --tk 0
echo $name
#

# results of 5way 1shot (vggnet)
 
# ----------------------------------------------------------

gpu=4
arch="resnet"
name=${N}w${K}s${Q}q${MB}m_protonet_${data}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

## to get test result, uncomment last line
CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
    --nw $N --ks $K --name $name --dset ${data} --maxe 150 \
    --arch $arch --gpuf 0.43 --qs $Q --mbsize $MB \
    --pr 1 --train 0 --vali 600 --tk 0
echo $name
#

# results of 5way 1shot (resnet)
# 
















#
