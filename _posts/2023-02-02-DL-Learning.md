---
layout: post
title: Convolutional Neural Networks

---

Convolutional Neural Network (CNN) includes an Overview, Examples, and Architectures.

---
## Convolutional Neural Network (CNN)
### [Softmax Activation function](https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78)
![](https://i.stack.imgur.com/0rewJ.png)
![](https://miro.medium.com/max/875/1*KvygqiInUpBzpknb-KVKJw.jpeg)


**Kaggle:** [https://www.kaggle.com/rkuo2000/mnist-cnn](https://www.kaggle.com/rkuo2000/mnist-cnn)<br>
[mnist_cnn.py](https://github.com/rkuo2000/tf/blob/master/mnist_cnn.py)<br>
```
from tensorflow.keras import models, layers, datasets
import matplotlib.pyplot as plt

# Load Dataset
mnist = datasets.mnist # MNIST datasets
(x_train_data, y_train),(x_test_data,y_test) = mnist.load_data()

# data normalization
x_train, x_test = x_train_data / 255.0, x_test_data / 255.0 
 
print('x_train shape:', x_train.shape)
print('train samples:', x_train.shape[0])
print('test samples:', x_test.shape[0])

# reshape for input
x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build Model
num_classes = 10 # 0~9

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(5, 5),activation='relu', padding='same',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary() 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train Model
epochs = 12 
batch_size = 128

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Save Model
models.save_model(model, 'models/mnist_cnn.h5')

# Evaluate Model
score = model.evaluate(x_train, y_train, verbose=0) 
print('\nTrain Accuracy:', score[1]) 
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest  Accuracy:', score[1])
print()

# Show Training History
keys=history.history.keys()
print(keys)

def show_train_history(hisData,train,test): 
    plt.plot(hisData.history[train])
    plt.plot(hisData.history[test])
    plt.title('Training History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
	
show_train_history(history, 'loss', 'val_loss')
show_train_history(history, 'accuracy', 'val_accuracy')
```

---

---
### Sound Digit CNN (語音數字分類)
**Dataset:** [台語0~9](https://www.kaggle.com/datasets/rkuo2000/sounddigittw)<br>
1. 使用手機錄音App(如: Voice Recorder) 將語音之數字存成.wav
2. 錄音時間長度= 1秒(語音錄得越長，訓練跟辨識的時間都較久)
3. 訓練資料集:每個數字錄20次 0_000.wav, 0_019.wav, 1_001.wav … 1_019.wav, …. 9_000.wav, … 9_019.wav
4. 測式資料集:每個數字錄2次 0_000.wav, 0_001.wav, 1_000.wav, 1_001.wav … 9_000.wav, 9_001.wav
5. 尋找手機檔案夾, 將其命名為SoundDigit, 底下有兩個目錄train (訓練集)跟test (測試集) 並壓縮成SoundDigit.zip
6. 上傳至你的Kaggle.com建立一個新的Dataset (例如: https://kaggle.com/rkuo2000/SoundDigitTW 是台語0~9)

**Kaggle:** [https://www.kaggle.com/code/rkuo2000/sounddigit-cnn](https://www.kaggle.com/code/rkuo2000/sounddigit-cnn/)<br>
* librosa to plot the waveform
```
for i in range(10):
    y, sr = librosa.load('Dataset/Train/'+str(i)+'_000.wav')
    plt.figure()
    plt.subplot(3,1,1)
    librosa.display.waveplot(y, sr=sr)
    plt.title('Waveform')
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sound_Digit_waveform.png?raw=true)
<br>
* extract MFCC spectorgram
```
x_train = list()
y_train = list()
(row, col) = (40,62)

for FILE in TRAIN_FILES:
    filename = FILE.replace('.3gp','.wav')
    mfcc = extract_feature('Dataset/Train/'+filename)
    # print(mfcc.shape)
    if (mfcc.shape[1]!=col):
        mfcc = np.resize(mfcc, (row,col))
    x_train.append(mfcc)
    y_train.append(FILE[0]) # first charactor of filename is the classname
    if FILE[2:5]=="000":
        print(mfcc.shape)
        display_mfcc(mfcc)  
```
![](https://github.com/rkuo2000/AI-course/blob/gh-pages/images/Sound_Digit_mel_freq_spectrogram.png?raw=true)

---


---
### Speech Commands (語音命令辨識)
**Dataset:** [Google Speech Commands](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands)<br>
Speech Commands: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"<br>

**Code:** [tensorflow/examples/speech_commands](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/qcnn-asr](https://www.kaggle.com/code/rkuo2000/qcnn-asr)<br>

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

![](https://1.bp.blogspot.com/-oNSfIOzO8ko/XO3BtHnUx0I/AAAAAAAAEKk/rJ2tHovGkzsyZnCbwVad-Q3ZBnwQmCFsgCEwYBhgL/s640/image3.png)

---
### CNN model comparison
![](https://miro.medium.com/max/1316/1*XoakexX4n9YSEalWxePqqw.png)


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
