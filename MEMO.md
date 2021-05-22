# Original SSD512

## Setup for VOC only

mkdir weights; cd weights
Download VOC512_FSSD_RESNEXT_137.pth resnext50_32x4d.pth

mkdir datasets; cd datasets
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

## Run for VOC

python eval_fssd_resnext_voc.py --trained_model=weights/VOC512_FSSD_RESNEXT_137.pth --voc_root=../datasets/VOCdevkit/
python train_fssd_resnext.py --dataset=VOC --dataset_root ../datasets/VOCdevkit/ --batch_size=4 --weight_prefix=VOC2007_
python pred_fssd_resnext.py --dataset=VOC --trained_model=weights/VOC512_FSSD_RESNEXT_137.pth data/example.jpg


## Detection

mkdir data/det
python convert.py # data/det/data.tsv
ln -s data.tsv data/det/train.tsv

python train.py --dataset_root data/det --batch_size=4 --weight_prefix=DET_
python predict.py --dataset_root data/det --trained_model=weights/DET_0.pth data/example.jpg
