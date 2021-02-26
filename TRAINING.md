
## Training with pretrained (Internet)
```
python train.py imagenette2-160/ --model resnet50 --amp --aa 'v0' --num-classes 5 --pretrained
```

## Training with pretrained (NO INTERNET)
```
python train.py imagenette2-160/ --model swsl_resnext50_32x4d --amp --aa 'v0' --num-classes 5  --load_state weights/swsl_resnext50_32x4d.pth
```

## Training from scratch
```
python train.py imagenette2-160/ --model resnet50 --amp --aa 'v0' --num-classes 5 --resume output/train/20210226-021234-resnet50-224/model_best.pth.tar
```

## Inference with trained model
```
python inference.py ./imagenette2-160/val/n01440764/ --model resnet50 --checkpoint ./output/train/20210225-144758-resnet50-224/model_best.pth.tar --num-classes 5 
```

# Useful stuff 

## list models with pretrained weights
avail_pretrained_models = timm.list_models(pretrained=True)

## BUG
cause of the Top5 Accuracy; minimum num_classes should be 5

## Dataset style
dataset/train/label1/abc.jpg
dataset/train/label2/234.jpg

dataset/val/label2/234.jpg
dataset/val/label1/234.jpg