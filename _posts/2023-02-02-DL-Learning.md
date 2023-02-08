---
layout: post
title: Convolutional Neural Networks

---

Convolutional Neural Network (CNN) includes an Overview, Examples, and Architectures.

---
## Convolutional Neural Network (CNN)
### softmax
![](https://i.stack.imgur.com/0rewJ.png)
![](https://miro.medium.com/max/875/1*KvygqiInUpBzpknb-KVKJw.jpeg)


**Kaggle:** [https://www.kaggle.com/rkuo2000/mnist-cnn](https://www.kaggle.com/rkuo2000/mnist-cnn)<br>
[mnist_cnn.py](https://github.com/rkuo2000/tf/blob/master/mnist_cnn.py)<br>

---
## [CNN architectures](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d)

* LeNet-5 (1998) [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
![](https://miro.medium.com/max/700/1*aQA7LuLJ2YfozSJa0pAO2Q.png)

* AlexNet (2012) [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
![](https://miro.medium.com/max/700/1*2DT1bjmvC-U-lrL7tpj6wg.png)

* VGG-16 (2014)
![](https://miro.medium.com/max/700/1*_vGloND6yyxFeFH5UyCDVg.png)

* Inception-v1 (2014)
![](https://miro.medium.com/max/700/1*KnTe9YGNopUMiRjlEr3b8w.png)

* Inception-v3 (2015)
![](https://miro.medium.com/max/700/1*ooVUXW6BIcoRdsF7kzkMwQ.png)

* ResNet-50 (2015)
![](https://miro.medium.com/max/700/1*zbDxCB-0QDAc4oUGVtg3xw.png)

* Xception (2016)
![](https://miro.medium.com/max/700/1*NNsznNjzULwqvSER8k5eDg.png)

* Inception-v4 (2016)
![](https://miro.medium.com/max/700/1*15sIUNOxqVyDPmGGqefyVw.png)

* Inception-ResNets (2016)
![](https://miro.medium.com/max/700/1*xpb6QFQ4IknSmxmgai8w-Q.png)

* DenseNet (2016)
![](https://miro.medium.com/max/1302/1*Cv2IqVWmiakP_boAJODKig.png)

* ResNeXt-50 (2017)
![](https://miro.medium.com/max/700/1*HelCJiQZEuwuKakRwDdGPw.png)

* EfficientNet (2019)
![](https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s640/image2.png)


---

## image classifaction

---


---
### Traffic Sign Classifier (交通號誌辨識)
**Dataset:** [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html)<br>
![](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/61e9ce225148f6519be6c034_GTSRB-0000000633-9ce3c5f6_Dki5Rsf.jpeg)

34 traffic signs, 39209 training images, 12630 test images<br>
**Kaggle:** [https://www.kaggle.com/rkuo2000/gtsrb-cnn](https://www.kaggle.com/rkuo2000/gtsrb-cnn)<br>

---
### Emotion Detection (情緒偵測)
**Dataset:** [FER-2013 (Facial Expression Recognition)](https://www.kaggle.com/datasets/msambare/fer2013)<br>
![](https://production-media.paperswithcode.com/datasets/FER2013-0000001434-01251bb8_415HDzL.jpg)

7 facial expression, 28709 training images, 7178 test images<br>
labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]<br>
**Kaggle:** [https://www.kaggle.com/rkuo2000/fer2013-cnn](https://www.kaggle.com/rkuo2000/fer2013-cnn)<br>

---
### Pneumonia Detection (肺炎偵測)
**Dataset:** [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)<br>
![](https://raw.githubusercontent.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning/master/demo/sample/sample.png)

**Kaggle:** [https://www.kaggle.com/rkuo2000/pneumonia-cnn](https://www.kaggle.com/rkuo2000/pneumonia-cnn)<br>

---
### COVID19 Detection (新冠肺炎偵測)
**Dataset:** [https://www.kaggle.com/bachrr/covid-chest-xray](https://www.kaggle.com/bachrr/covid-chest-xray)<br>
![](https://i.imgur.com/jZqpV51.png)

**Kaggle:**<br>
* [https://www.kaggle.com/rkuo2000/covid19-vgg16](https://www.kaggle.com/rkuo2000/covid19-vgg16)
* [https://www.kaggle.com/rkuo2000/skin-lesion-cnn](https://www.kaggle.com/rkuo2000/skin-lesion-cnn)

---
### FaceMask Classification (人臉口罩辨識)
**Dataset:** [Face Mask ~12K Images dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/facemask_12k_dataset.png?raw=true)

**Kaggle:** [https://www.kaggle.com/rkuo2000/facemask-cnn](https://www.kaggle.com/rkuo2000/facemask-cnn)<br>

---
### Garbage Classification (垃圾分類)
**Dataset:** https://www.kaggle.com/asdasdasasdas/garbage-classification (42MB)<br>
<img widtih="50%" height="50%" src="https://miro.medium.com/max/2920/1*mJipx8yxeI_JW36jDAuM9A.png">

6 categories : cardboard(403), glass(501), metal(410), paper (594), plastic(482), trash(137)<br>

**Kaggle:** [https://www.kaggle.com/rkuo2000/garbage-cnn](https://www.kaggle.com/rkuo2000/garbage-cnn)<br>

---
### Food Classification  (食物分類)
**Dataset:** [Food-11](https://mmspg.epfl.ch/downloads/food-image-datasets/)<br>
![](https://929687.smushcdn.com/2633864/wp-content/uploads/2019/06/fine_tuning_keras_food11.jpg?lossy=1&strip=1&webp=1)
The dataset consists of 16,643 images belonging to 11 major food categories:<br>
* Bread (1724 images)
* Dairy product (721 images)
* Dessert (2,500 images)
* Egg (1,648 images)
* Fried food (1,461images)
* Meat (2,206 images)
* Noodles/pasta (734 images)
* Rice (472 images)
* Seafood (1,505 images)
* Soup (2,500 images)
* Vegetable/fruit (1,172 images)

**Kaggle:** [https://www.kaggle.com/rkuo2000/food11-classification](https://www.kaggle.com/rkuo2000/food11-classification)<br>

---
### Mango Classification (芒果分類)
**Dataset:** [台灣高經濟作物 - 愛文芒果影像辨識正式賽](https://aidea-web.tw/aicup_mango)<br>
**Kaggle:** <br>
* [https://www.kaggle.com/rkuo2000/mango-classification](https://www.kaggle.com/rkuo2000/mango-classification)
* [https://www.kaggle.com/rkuo2000/mango-efficientnet](https://www.kaggle.com/rkuo2000/mango-efficientnet)

---
## Transer Learning

### Birds Classification (鳥類分類)
**Dataset:** [https://www.kaggle.com/rkuo2000/birds2](https://www.kaggle.com/rkuo2000/birds2)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/birds_dataset.png?raw=true)

---
### Animes Classification (卡通人物分類)
**Dataset:** [https://www.kaggle.com/datasets/rkuo2000/animes](https://www.kaggle.com/datasets/rkuo2000/animes)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/animes_dataset.png?raw=true)

**Kaggle:** [https://www.kaggle.com/rkuo2000/anime-classification](https://www.kaggle.com/rkuo2000/anime-classification)<br>

---
### Worms Classification(害蟲分類)
**Dataset:** [worms4](https://www.kaggle.com/datasets/rkuo2000/worms4)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/worms4_dataset.png?raw=true)

用Google搜尋照片, 下載各20/30張照片，放入資料夾worms後，壓縮成worms.zip, 再上傳Kaggle.com/datasets<br>

**Kaggle:** [https://www.kaggle.com/rkuo2000/worms-classification](https://www.kaggle.com/rkuo2000/worms-classification)<br>

---
### Railway Track Fault Detection (鐵軌故障偵測)
**Dataset:** [Railway Track Fault Detection](https://www.kaggle.com/salmaneunus/railway-track-fault-detection)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/railtrack-resnet50v2](https://www.kaggle.com/code/rkuo2000/railtrack-resnet50v2)<br>
```
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import models, layers

base_model=ResNet50V2(input_shape=input_shape,weights='imagenet',include_top=False) 
base_model.trainable = False # freeze the base model (for transfer learning)

# add Fully-Connected Layers to Model
x=base_model.output
x=layers.GlobalAveragePooling2D()(x)
x=layers.Dense(128,activation='relu')(x)  # FC layer 
preds=layers.Dense(num_classes,activation='softmax')(x) #final layer with softmax activation

model=models.Model(inputs=base_model.input,outputs=preds)
model.summary()
```
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/railtrack-efficientnet](https://www.kaggle.com/code/rkuo2000/railtrack-efficientnet)<br>
```
import efficientnet.tfkeras as efn
from tensorflow.keras import models, layers, optimizers, regularizers, callbacks

base_model = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)
base_model.trainable = False # freeze the base model (for transfer learning)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=out)

model.summary()
```

---
### Skin Lesion Classification (皮膚病變分類)
**Dataset:*8 [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)<br>
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/ham10000_dataset.png?raw=true)

7 types of lesions : (picture = 600x450)<br>
* Actinic Keratoses (光化角化病)
* Basal Cell Carcinoma (基底細胞癌)
* Benign Keratosis (良性角化病)
* Dermatofibroma (皮膚纖維瘤)
* Malignant Melanoma (惡性黑色素瘤)
* Melanocytic Nevi (黑素細胞痣)
* Vascular Lesions (血管病變)
<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/skin-lesion-classification](https://www.kaggle.com/code/rkuo2000/skin-lesion-classification)<br>

* assign base_model
```
#base_model=applications.MobileNetV2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.InceptionV3(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.ResNet50V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.ResNet101V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.ResNet152V2(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.DenseNet121(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.DenseNet169(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.DenseNet201(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.NASNetMobile(input_shape=(224,224,3), weights='imagenet',include_top=False)
#base_model=applications.NASNetLarge(input_shape=(331,331,3), weights='imagenet',include_top=False)
```

* import EfficentNet model
```
import efficientnet.tfkeras as efn
base_model = efn.EfficientNetB7(input_shape=(224,224,3), weights='imagenet', include_top=False)
```

<br>
<br>






---

## Archive

---
## Pytorch Image Models (https://rwightman.github.io/pytorch-image-models/models/)

### Big Transfer ResNetV2 (BiT) [resnetv2.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnetv2.py)
* Paper: [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)
* Code: [https://github.com/google-research/big_transfer](https://github.com/google-research/big_transfer)

### Cross-Stage Partial Networks [cspnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cspnet.py)
* Paper: [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)
* Code: [https://github.com/WongKinYiu/CrossStagePartialNetworks](https://github.com/WongKinYiu/CrossStagePartialNetworks)

### DenseNet [densenet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/densenet.py)
* Paper: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
* Code: [https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

### DLA [dla.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/dla.py)
* Paper: [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484)
* Code: [https://github.com/ucbdrive/dla](https://github.com/ucbdrive/dla)

### Dual-Path Networks [dpn.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/dpn.py)
* Paper: [Dual Path Networks](https://arxiv.org/abs/1707.01629)
* Code: [https://github.com/rwightman/pytorch-dpn-pretrained](https://github.com/rwightman/pytorch-dpn-pretrained)

### GPU-Efficient Networks [byobnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py)
* Paper: [Neural Architecture Design for GPU-Efficient Networks](https://arxiv.org/abs/2006.14090)
* Code: [https://github.com/idstcv/GPU-Efficient-Networks](https://github.com/idstcv/GPU-Efficient-Networks)

### HRNet [hrnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py)
* Paper: [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919)
* Code: [https://github.com/HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)

### Inception-V3 [inception_v3.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v3.py)
* Paper: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
* Code: [https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

### Inception-V4 [inception_v4.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v4.py)
* Paper: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### Inception-ResNet-V2 [inception_resnet_v2.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_resnet_v2.py)
* Paper: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### NASNet-A [nasnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nasnet.py)
* Papers: [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### PNasNet-5 [pnasnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/pnasnet.py)
* Papers: [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### EfficientNet [efficientnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py)
* Paper: [EfficientNet (B0-B7)](https://arxiv.org/abs/1905.11946)
  - [EfficientNet NoisyStudent (B0-B7, L2)](https://arxiv.org/abs/1911.04252)
  - [EfficientNet AdvProp (B0-B8)](https://arxiv.org/abs/1911.09665)
* Code: [https://github.com/rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)

### MobileNet-V3 [mobilenetv3.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py)
* Paper: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* Code: [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

### RegNet [regnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py)
* Paper: [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
* Code: [https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py](https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py)

### RepVGG [byobnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py)
* Paper: [Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
* Code: [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)

### ResNet, ResNeXt [resnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py)
**ResNet (V1B)**<br>
* Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* Code: [https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

**ResNeXt**<br>
* Paper: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
* Code: [https://github.com/pytorch/vision/tree/master/torchvision/models](https://github.com/pytorch/vision/tree/master/torchvision/models)

**ECAResNet (ECA-Net)**<br>
* Paper: [ECA-Net: Efficient Channel Attention for Deep CNN](https://arxiv.org/abs/1910.03151v4)
* Code: [https://github.com/BangguWu/ECANet](https://github.com/BangguWu/ECANet)

### Res2Net [res2net.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py)
* Paper: [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)
* Code: [https://github.com/gasvn/Res2Net](https://github.com/gasvn/Res2Net)

### ResNeSt [resnest.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnest.py)
* Paper: [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)
* Code: [https://github.com/zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)

### ReXNet [rexnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/rexnet.py)
* Paper: [ReXNet: Diminishing Representational Bottleneck on CNN](https://arxiv.org/abs/2007.00992)
* Code: [https://github.com/clovaai/rexnet](https://github.com/clovaai/rexnet)

### Selective-Kernel Networks [sknet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/sknet.py)
* Paper: [Selective-Kernel Networks](https://arxiv.org/abs/1903.06586)
* Code: [https://github.com/implus/SKNet](https://github.com/implus/SKNet), [https://github.com/clovaai/assembled-cnn](https://github.com/clovaai/assembled-cnn)

### SelecSLS [selecsls.py]
* Paper: [XNect: Real-time Multi-Person 3D Motion Capture with a Single RGB Camera](https://arxiv.org/abs/1907.00837)
* Code: [https://github.com/mehtadushy/SelecSLS-Pytorch](https://github.com/mehtadushy/SelecSLS-Pytorch)

### Squeeze-and-Excitation Networks [senet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/senet.py)
* Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### TResNet [tresnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/tresnet.py)
* Paper: [TResNet: High Performance GPU-Dedicated Architecture](https://arxiv.org/abs/2003.13630)
* Code: [https://github.com/mrT23/TResNet](https://github.com/mrT23/TResNet)

### VGG [vgg.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py)
* Paper: [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
* Code: [https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)

### Vision Transformer [vision_transformer.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
* Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* Code: [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

### VovNet V2 and V1 [vovnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py)
* Paper: [CenterMask : Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667)
* Code: [https://github.com/youngwanLEE/vovnet-detectron2](https://github.com/youngwanLEE/vovnet-detectron2)

### Xception [xception.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/xception.py)
* Paper: [Xception: Deep Learning with Depthwise Separable Convolutions(https://arxiv.org/abs/1610.02357)
* Code: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

### Xception (Modified Aligned, Gluon) [gluon_xception.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/gluon_xception.py)
* Paper: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
* Code: [https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo), [https://github.com/jfzhang95/pytorch-deeplab-xception/](https://github.com/jfzhang95/pytorch-deeplab-xception/)

### Xception (Modified Aligned, TF) [aligned_xception.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/aligned_xception.py)
* Paper: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
* Code: [https://github.com/tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
