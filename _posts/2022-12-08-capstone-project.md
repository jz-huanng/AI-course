---
layout: post
title: capstone project
author: [pride_number_]
category: [lecture]
tags: [jekyll, ai]
---




<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov8" target="_blank">
      <img width="850" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>
 
 <div align="left">

### 專題描述
---



### 訓練資料
---


**蒐集資料**<br><br>

**手動添加標籤**<br><br>

資料標註工具參考[郭子仁老師的網頁](https://rkuo2000.github.io/AI-course/lecture/2022/10/13/Object-Detection-Exercises.html)<br><br>

以這張圖為例，可以看到圖中兩台車一台車是停在格子裡，一台車則是隨意停靠路邊。分別對左邊的貼標籤```legal```以及右邊的```illegal```<br>
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/labels.png?raw=true)<br>

**kaggle的部分**
 <br>
 
 ```
 git clone https://github.com/jz-huanng/yolov5
 
 ```
 
 在資料夾data放進images 和 labels
 ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/directory.png?raw=true)

**custom yaml file**

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: ../images  # dataset root dir
train: train  # train images (relative to 'path') 128 images
val: val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 2  # number of classes
names: ['legal', 'illegal']  # class names
```

注意的是yaml檔的位址應該和train.py同一層資料夾。<br>

放錯位址造成的結果:
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/yaml_in_wrong_sdirectory.png?raw=true)


### 辨識結果
---

***train***
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/train.png?raw=true)

image num:15<br>
epoch 100<br>
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/num15.png?raw=true)

fine-tune<br>
image num:5<br>
epoch 80<br>
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/fine-tune.png?raw=true)


### kaggle實作
---

[kaggle:parking detection](https://www.kaggle.com/code/ulysses1103/parked-detection)<br>

### 參考資料
---

[Image Annotation](https://rkuo2000.github.io/AI-course/lecture/2022/10/13/Object-Detection-Exercises.html
)<br><br>

[yolov5](https://github.com/ultralytics/yolov5)

<br>

>quote




<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
### ARCHIVE棄存
---
我們希望藉由yolo實現停車格空位的辨識<br>
我們要客製化自己的訓練資料，同時**避免偵測汽車、機車等非預期結果**。理想上是偵測停車格上是否有汽機車佔位。<br>
每一次框選都要**框住白色方框**，在依據是否停有車來選擇不同的標籤```empty``` 或 ```parked ```。有些照片中**機車並非出現在白色方框內不應該被標註出來**。<br><br>


**imperfect outcome**

![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/bad_outcome.png?raw=true)

結果不理想原因應該在於蒐集資料的角度


*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

