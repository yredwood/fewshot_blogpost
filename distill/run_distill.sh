gpu=6
A=1
sarch="simple"
tarch="vgg"
config="cl_cifar10"  #for learn
#config="cl_miniImagenet"
tname=Learner_${config}_${tarch}
sname=Learner_${config}_${sarch}


name=Distill_${config}_${tarch}2${sarch}3
log="../models/${name}/log.txt"
mkdir -p ../models/${name}

#echo $log
#exec &> >(tee -a "$log")

# ------------------------------------------------------------------------
# first you need to pretrain the network
CUDA_VISIBLE_DEVICES=$gpu python -u distilling.py --maxe 200 --tarch $tarch \
    --sarch $sarch --tname $tname --sname $sname \
    --config $config --bs 128 --gpuf 0.91 --name $name --aug=$A  \
    --pr 1 --train 0 --vali 600  # test code


## ------------------------------------------------------------------------
#A=0
#P=1
#config="miniImagenet" # for metalearn
#name=AdaptNet_${config}_aug${A}_fix${F}_U${U}_${M}${N}${K}
##CUDA_VISIBLE_DEVICES=$gpu python -u metalearning.py --gpuf 0.91 --pr $P \
##    --config $config --qs $Q --nw $N --ks $K --name $name \
##    --mbsize $M --aug $A --fix_univ $F --maxe 20 --use_adapt $U \
#
