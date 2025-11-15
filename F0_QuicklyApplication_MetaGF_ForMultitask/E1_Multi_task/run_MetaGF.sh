cd ./models/
logfir=logs1212emaratio/
mkdir -p $logfir
dataroot="../../data/nyuv2"
#dataroot="/media/sunyi/E/nyuv2"
weight=equal
seed=0
metalr=0.1
ema=0.9
auxlr=0.5
using_scale=0
weightdecay=0
cudadevice=cuda:0
inverse=0
MetaGFstartEpoch=100
usinglosslandscape=0
CONSTANTSHARELAYER=0
SCALINGvalue=1e-3
python -u model_segnet_mt_ECCV.py  --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight  --metalr $metalr --device $cudadevice  --ema $ema --auxlr $auxlr > $logfir/Metadevice$cudadevice-metalr$metalr-ema$ema-auxlr$auxlr-weightdecay$weightdecay.log