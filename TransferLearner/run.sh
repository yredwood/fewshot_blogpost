config="miniimg64"
aug=1

gpu=4
arch="simple"
name=learner_${config}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

## first you need to pretrain the network
#CUDA_VISIBLE_DEVICES=$gpu python -u pretrain.py --lr 1e-1 \
#    --config $config --arch $arch --bs 128 --gpuf 0.91 --name $name \
#    --pr 1 --train 0 --vali 600  # test code
#
## results
#
#
#
#
####--------------------------------------------------------------
gpu=4
arch="vggnet"
name=learner_${config}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

## first you need to pretrain the network
#CUDA_VISIBLE_DEVICES=$gpu python -u pretrain.py --lr 1e-1 \
#    --config $config --arch $arch --bs 128 --gpuf 0.93 --name $name \
#    --pr 1 --train 0 --vali 600  # test code

# results
#
####--------------------------------------------------------------
gpu=3
arch="resnet"
name=learner_${config}_${arch}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}
##
## first you need to pretrain the network
CUDA_VISIBLE_DEVICES=$gpu python -u pretrain.py --lr 1e-1 --maxe 150 \
    --config $config --arch $arch --bs 128 --gpuf 0.91 --name $name --aug $aug \
#    --pr 1 --train 0 --vali 600  # test code
#
### results
