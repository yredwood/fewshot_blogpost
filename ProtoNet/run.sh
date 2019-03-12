gpu=4
N=5 # n-way 
K=15 # k-shot
data="miniImagenet"
name=${W}w${S}s_protonet_manyshot_${data}
log="../models/${name}/log.txt"
mkdir -p ../models/${name}


######  GUIDE  ######
# K : training shots 
# tk : test shots


# to get test result, uncomment last line
#echo $log
#exec &> >(tee -a "$log")
CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
    --nw $N --ks $K --name $name --dset ${data} --lr 1e-3 \
#    --pr 1 --train 0 --vali 600 --tk 150

