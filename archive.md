---
layout: archive
type: years
title: Archive
permalink: /archive/
---




## Lesson

https://rkuo2000.github.io/AI-course/

[lesson 4]2022-10-06-Image-Classification

Transer Learning

https://www.kaggle.com/datasets/rkuo2000/animes
	

### RL-gym
---
pip install pyglet==1.5.27
pip install stable_baselines3[extra]
pip install gym[all]
pip install autorom[accept-rom-license]
git clone https://github.com/rkuo2000/RL-gym
cd RL-gym

pip install box2d-py 

cd RL-gym
cd sb3

python train.py LunarLander-v2 640000
python enjoy.py LunarLander-v2
python enjoy_gif.py LunarLander-v2


python train.py lunar_Lander-v2 640000

python train.py LunarLander 640000

AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLander'

raise error.UnregisteredEnv("No registered env with id: {}".format(id))
gym.error.UnregisteredEnv: No registered env with id: lunar_Lander-v2

A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\stable_baselines3\common\save_util.py:166: UserWarning: Could not deserialize object lr_schedule. Consider using `custom_objects` argument to replace this object.
  warnings.warn(

### facemet
---

[message]too many values to unpack (expected 2)<br>
 
[reference]https://www.pythonpool.com/valueerror-too-many-values-to-unpack-expected-2-solved/
 
[solution]<br>
 
aligned = []
names = []
for x, y in loader:
    x_aligned, prob ,*unpack= mtcnn(x, return_prob=True)
       if x_aligned is not None:
         #print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

print(len(aligned))
print(names)


### MMPose
---

鄭濬緯<br>
<br>
!pip install scipy==1.3.1<br>
<br>
rkuo2000<br>
rkuo2000：我殺掉舊的, 重新按github的重新跑<br> 
rkuo2000：應該是安裝問題 <br>
rkuo2000：pip install -e . 就可以了<> 
	







*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
