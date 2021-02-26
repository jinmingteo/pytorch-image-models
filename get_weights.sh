mkdir -p weights

# tf_efficientnet_b4_ns
wget -O weights/tf_efficientnet_b4_ns.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth

# tf_efficientnet_b3_ns
wget -O weights/tf_efficientnet_b3_ns.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ns-9d44bf68.pth

# resnet50
wget -O weights/resnet50.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth

# swsl_resnext50_32x4d
wget -O weights/swsl_resnext50_32x4d.pth https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth

# swsl_resnet18
wget -O weights/swsl_resnet18.pth  https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth

# ssl_resnet18
wget -O weights/ssl_resnet18.pth https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth

# ssl_resnet50
wget -O weights/ssl_resnet50.pth https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth
