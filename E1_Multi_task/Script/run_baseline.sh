cd ../models/
logfir=logs_nyuv2
mkdir -p $logfir
dataroot="/home/sunyi/nyuv2_10"
weight=equal
seed=0
cudadevice=0
python -u model_segnet_mtan.py  --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight  --device $cudadevice > $logfir/device$cudadevice-$weight.log
