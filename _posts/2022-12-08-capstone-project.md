---
layout: post
title: capstone project
author: [pride_number_]
category: [lecture]
tags: [jekyll, ai]
---





# UNDER CONSTRUCTION..
<br><br><br><br><br><br>
### method
- classifacation or object detection?
- opencv module or neural network models(yolo)?
- view:aerial or walker?

### something new!
- pytorch detectron
- Meta SAM
- RT-DETR
<br><br><br><br><br><br>



---
### 專題描述
---

[kaggle:parking detection](https://www.kaggle.com/code/ulysses1103/parked-detection)<br>


### 建構自定義資料集
---


**蒐集資料**<br><br>

**手動添加標籤**<br><br>

以這張圖為例，可以看到圖中兩台車一台車是停在格子裡，一台車則是隨意停靠路邊。分別對左邊的貼標籤```legal```以及右邊的```illegal```<br>
![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/labels.png?raw=true)<br>

**clone github的倉庫**
 
 ```
 git clone https://github.com/jz-huanng/yolov5
 ```

**custom yaml file**

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]

path: ../yolov5/data/Tofen_parking/images  # dataset root dir
train: train  # train images (relative to 'path') 128 images
val: val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 2  # number of classes
names: ['legal', 'illegal']  # class names
```

### 開始訓練
---

run on kaggle<br>
[kaggle:parking detection](https://www.kaggle.com/code/ulysses1103/parked-detection)<br>
<br>
<br>



### 用測試集驗證模型
---

#### 版本展示

| image num | epoch | results | description |see results on kaggle |
| --: | -- | -- | --: | --|
| 15 | 100 |  ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/num15.png?raw=true) | labelled wrong<br>這些應該被貼上```illegal```而不適```legal``` |version 8 | 
| 15(fine-tuned) | 80 | ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/fine-tune.png?raw=true)<br>![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/labels2.png?raw=true) | labelled imprfect 但是不是機器學到方格? | version 9| 
| 15(fine-tuned) | 80 | ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/no_result.png?raw=true)| 沒有辨識出來 | version 9| 
| 15(fine-tuned) | 160 | ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/epoch160.png?raw=true)| 目前下來唯一一張有辨識到illegal | version 10| 
| 15(fine-tuned) | 160 | ![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/epoch160_2.png?raw=true)| 和epoch 80結果一樣 | version 10| 
|44|80|![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/version11.png?raw=true)|訓練資料變多不過epoch數只有80練不起來|version 11|
|44|160|![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/parking-detection/version13.png?raw=true)|幾乎成功!|version13|

### 最終成果
---

第一次訓練遇到幾個障礙，一是這次並沒有辦法辨識出違規停車的部分，而且似乎yolo是練成**單純辨識機車**。<br><br>
這讓我們很挫折。第二次訓練，重新蒐集**充足**的資料，因為有可能是**違規停車**的不夠多。而在第二次的標註中，也找到訣竅：除了框住白色方格，如果它是違規停車的話，**框住穿過穿過的黃線(或者紅線)**，整體看起來會是長方形框。<br><br>
而第二次的確進步不少，雖然有些還是沒有辨識到。<<br>另外缺點是：訓練資料44張，相比第一次15張，而且epoch要來到160才會辨識到，訓練時長2小時。<br><br>
#### 後續想法：
更精準的資料蒐集，更預設結果的標註，也許圖像資料壓縮可以讓電腦訓練的過程縮短。<br>

<br>
To be continued..

### 討論

經過多次的修改與嘗試大概想到**如何練好模型** 

<br>




<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
### ARCHIVE棄存
---

**imperfect outcome**

![](https://github.com/jz-huanng/AI-course/blob/gh-pages/images2/bad_outcome.png?raw=true)

結果不理想原因應該在於蒐集資料的角度


*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

