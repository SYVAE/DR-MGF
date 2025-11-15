cd ../models/
logfir=logs20250305emaratio
mkdir -p $logfir
dataroot="/home/sunyi/nyuv2_10"
#dataroot="/media/sunyi/E/nyuv2"
weight=equal
seed=0
metalr=0.1
ema=0.9
auxlr=0.5
using_scale=0
weightdecay=0
cudadevice=0
inverse=0
MetaGFstartEpoch=100
usinglosslandscape=0
CONSTANTSHARELAYER=0
SCALINGvalue=1e-3

seed=0
python -u model_segnet_mt_DRMGF.py --SCALINGvalue $SCALINGvalue --CONSTANTSHARELAYER $CONSTANTSHARELAYER --usinglosslandscape $usinglosslandscape --MetaGFstartEpoch $MetaGFstartEpoch --weightdecay $weightdecay --inverse $inverse --usinglossscale  $using_scale  --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight  --metalr $metalr --device $cudadevice  --ema $ema --auxlr $auxlr > $logfir/Seed$seed-device$cudadevice-SCALINGvalue$SCALINGvalue-$weight-metalr$metalr-ema$ema-auxlr$auxlr-scale$using_scale-inverse$inverse-weightdecay$weightdecay--MetaGFstartEpoch$MetaGFstartEpoch--usinglosslandscape$usinglosslandscape--CONSTANTSHARELAYER$CONSTANTSHARELAYER.log
