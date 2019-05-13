N=5
K=1
Q=15

#P="../models/Metalearn_miniImagenet_simple_120500_Pretrained_1e-3_1/miniImagenet.ckpt"
#P="none"
C="miniImagenet"
Ar="simple"
MAr="proto"
cluster_npy="tsklist_random_large20w.npy"

for ((i=100;i<=149;i++)); do
    pnt=$i
    #gpu=$(( $pnt - 6 ))
    #gpu=$pnt
    gpu=5
    echo $gpu

    name=Metalearn_${C}_${Ar}_120500_Pretrained_${cluster_npy}

    CUDA_VISIBLE_DEVICES=$gpu python make_test_features.py \
        --qs 15 --nw 5 --ks $K \
        --gpuf 0.44 --config $C \
        --arch $Ar --train 0 --vali 600 \
        --meta_arch $MAr --name $name --pn $pnt \
        --cl clusters/$cluster_npy
done
