---
layout: post
title: capstone project
author: [Charles Huang]
category: [Lecture]
tags: [jekyll, ai]
---

---

### 專題描述
---

我們希望藉由yolo實現停車格空位的辨識<br>

### 訓練資料
---
我們要客製化自己的訓練資料，同時**避免偵測汽車、機車等非預期結果**。理想上是偵測停車格上是否有汽機車佔位。<br>

**蒐集資料**<br><br>
首先我們得先準備足夠的訓練資料，拍攝地點選擇國立臺灣海洋大學，找尋適當的角度進行拍攝，接著標註沒一筆數據。<br><br>

**手動添加標籤**<br><br>

每一次框選都要**框住白色方框**，在依據是否停有車來選擇不同的標籤```empty``` 或 ```parked ```。有些照片中**機車並非出現在白色方框內不應該被標註出來**。<br><br>

資料標註工具參考[郭子仁老師的網頁](https://rkuo2000.github.io/AI-course/lecture/2022/10/13/Object-Detection-Exercises.html)<br><br>
 
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

注意!這份檔案存放不在data,而是在上一層

放錯位址:
![]()

**程式碼的部分**


<br><br><br>

### 辨識結果
---

[kaggle連結](https://www.kaggle.com/ulysses1103/parked-detection)

### 初次結果

**imperfect outcome**

![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/bad_outcome.png?raw=true)

結果不理想原因應該在於蒐集資料的角度

---

### 參考資料
---

[Image Annotation](https://rkuo2000.github.io/AI-course/lecture/2022/10/13/Object-Detection-Exercises.html
)<br><br>

[yolov5](https://github.com/ultralytics/yolov5)

<br>

>quote





*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

