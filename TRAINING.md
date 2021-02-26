```
python train.py imagenette2-160/ --model resnet50 --amp --aa 'v0' --num-classes 5 --pretrained
```

```
python inference.py ./imagenette2-160/val/n01440764/ --model resnet50 --checkpoint ./output/train/20210225-144758-resnet50-224/model_best.pth.tar --num-classes 5 
```
