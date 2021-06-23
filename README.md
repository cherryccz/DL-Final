# DL-Final
Image translation on face via pretrained stylegan2 model

**整体流程**：

![image](https://user-images.githubusercontent.com/63801551/123090714-0cfdbc80-d45b-11eb-9855-618d285bc7d5.png)


**datasets**：

![image](https://user-images.githubusercontent.com/63801551/123090894-433b3c00-d45b-11eb-94e9-0843cb227322.png)



下载stylegan2-ada代码


```python
!git clone https://github.com/dvschultz/stylegan2-ada
```




```python
%cd stylegan2-ada
```


```python
%tensorflow_version 1.x
!pip install typer
```


```python
!git clone https://github.com/wkentaro/gdown
%cd gdown
!pip install gdown
```

下载stylegan2-ffhq预训练网络


```python
!gdown https://drive.google.com/uc?id=1wHv4hjRkS6E2SZ6RCn1seLZjZAfy5ETo
```



# **model training**

数据集准备工作（可跳过）


```python
%run align_images.py /file_path /disney_images
```


```python
%run dataset_tool.py create_from_images_raw /datasets/disney /disney_images
```


```python
from IPython.display import Image 
```



### Fine tune


```python
!python stylegan2-ada/train.py --gpus=2 --outdir=results --data=datasets/Disney --resume=stylegan2-ffhq-config-f.pkl --metrics=None --kimg=10000 --snap=50
!python stylegan2-ada/train.py --gpus=2 --outdir=results --data=datasets/portrait --resume=stylegan2-ffhq-config-f.pkl --metrics=None --kimg=10000 --snap=50
!python stylegan2-ada/train.py --gpus=2 --outdir=results --data=datasets/GQJ --resume=stylegan2-ffhq-config-f.pkl --metrics=None --kimg=10000 --snap=50
```

![image](https://user-images.githubusercontent.com/63801551/123090919-4e8e6780-d45b-11eb-834e-d7358fc6ade8.png)



# **model blending**


## face2cartoon

```python
!gdown https://drive.google.com/uc?id=1frCcmHS8s15fFJvY0ctEISBprilgAMh6 #下载卡通模型
```


```python
!python blend_models.py stylegan2-ffhq-config-f.pkl FFHQ-Cartoons.pkl 16 --output-pkl="blended16.pkl"
```


blended models:


*   ffhq-cartoon-blended-4: https://drive.google.com/file/d/13JsjrUz7RApWwTJ58wCy9iv38EOtqlU6/view?usp=sharing
*   ffhq-cartoon-blended-16:https://drive.google.com/file/d/13DKF8yujBlWlXeArx6DBoGU6cg9tpTTu/view?usp=sharing
*   ffhq-cartoon-blended-64:https://drive.google.com/file/d/13NVXnSezNwJAcmyRLAtlF3nGdxKfKOSA/view?usp=sharing



## face2portait


```python
!gdown https://drive.google.com/uc?id=1k56xV4Cu2TPxqYZy8XD8Z3jK5EZJiw99 #下载metface模型
```


```python
!python blend_models.py stylegan2-ffhq-config-f.pkl metfaces.pkl 4 --output-pkl=ffhq-metfaces-blended-4.pkl
!python blend_models.py stylegan2-ffhq-config-f.pkl metfaces.pkl 16 --output-pkl=ffhq-metfaces-blended-16.pkl  
!python blend_models.py stylegan2-ffhq-config-f.pkl metfaces.pkl 64 --output-pkl=ffhq-metfaces-blended-64.pkl
```

blended models:


*   ffhq-metfaces-blended-4: https://drive.google.com/file/d/1-DUjcS4mpuY2LOSlyZXRYlGGWA2UFEfn/view?usp=sharing
*   ffhq-metfaces-blended-16:https://drive.google.com/file/d/1ONbf-mGqRTuFw551UWUDBCvDy-SZA4-8/view?usp=sharing
*   ffhq-metfaces-blended-64:https://drive.google.com/file/d/13RUaSF7dpBXFGkMUSYht0FvS1Jj_EYMI/view?usp=sharing

some results:

![image](https://user-images.githubusercontent.com/63801551/123091394-deccac80-d45b-11eb-9473-49800bd3efd5.png)



## multi-domian


```python
!python blend_models.py blended16.pkl metface.pkl 64 --output-pkl=ffhq-cartoon16-metfaces64.pkl
```

some results:

![image](https://user-images.githubusercontent.com/63801551/123091461-f310a980-d45b-11eb-8fa3-464a840dd579.png)


# compute FID & LPIPS


## genenrate image


```python
pip install opensimplex
```


```python
import random
seeds = random.sample(range(1,5000), 2000)
```


```python
!python stylegan2-ada/generate.py --outdir=output/cartoon --trunc=1 --seeds=2000  --network=FFHQ-CartoonsAlignedHQ36v2.pkl
!python stylegan2-ada/generate.py --outdir=output/blend4 --trunc=1 --seeds=2000  --network=blended4.pkl
!python stylegan2-ada/generate.py --outdir=output/blend16 --trunc=1 --seeds=2000  --network=blended16.pkl
!python stylegan2-ada/generate.py --outdir=output/blend64 --trunc=1 --seeds=2000  --network=blended64.pkl
```


```python
!python stylegan2-ada/generate.py --outdir=output/metface --trunc=1 --seeds=2000  --network=metfaces.pkl
!python stylegan2-ada/generate.py --outdir=output/metface4 --trunc=1 --seeds=2000  --network=ffhq-metfaces-blended-4.pkl
!python stylegan2-ada/generate.py --outdir=output/metface16 --trunc=1 --seeds=2000  --network=ffhq-metfaces-blended-16.pkl
!python stylegan2-ada/generate.py --outdir=output/metface64 --trunc=1 --seeds=2000  --network=ffhq-metfaces-blended-64.pkl
```


```python
!python stylegan2-ada/generate.py --outdir=output/cartoon16-metface64 --trunc=1 --seeds=2000  --network=ffhq-cartoon16-metfaces64.pkl
```

## compute

[FID](https://github.com/bioinf-jku/TTUR/blob/master)   [LPIPS](https://github.com/richzhang/PerceptualSimilarity/blob/master)

```python
 # need inception-2015-12-05.tgz
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/cartoon 
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/blend4
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/blend16 
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/blend64

!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/metface
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/metface4
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/metface16 
!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/metface64 

!python fid.py --inception=/nfsshare/home/dl11/stylegan2-ada output/ffhq output/cartoon16-metface64 
```


```python
!python lpips_2dirs.py -d0 output/ffhq -d1 output/cartoon -o output/lpips_ffhq_cartoon.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/blend4 -o output/lpips_ffhq_blend4.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/blend16 -o output/lpips_ffhq_blend16.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/blend64 -o output/lpips_ffhq_blend64.txt

!python lpips_2dirs.py -d0 output/ffhq -d1 output/metface -o output/lpips_ffhq_metface.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/metface4 -o output/lpips_ffhq_metface4.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/metface16 -o output/lpips_ffhq_metface16.txt
!python lpips_2dirs.py -d0 output/ffhq -d1 output/metface64 -o output/lpips_ffhq_metface64.txt
```


```python
import numpy as np

filename='output/lpips_ffhq_metface4.txt'
with open(filename, 'r') as f:
    lines = f.read().splitlines()
data = []    
for line in lines:
    d = {}
    _, n = line.split(': ')
    data.append(float(n))
print(np.mean(data))
```


```python

```

result:

|         | **disney** |           | **portrait** |           |
| ------- | ---------- | --------- | ------------ | --------- |
|         | *FID*      | *LPIPS*S  | *FID*        | *LPIPS*S  |
| FT      | 77.400     | 0.561     | 116.702      | 0.734     |
| FT+LS4  | 70.588     | 0.530     | 108.462      | 0.725     |
| FT+LS16 | 49.599     | 0.433     | 91.227       | 0.679     |
| FT+LS64 | **20.456** | **0.256** | **57.652**   | **0.549** |



# **ganspace**


```python
%cd /content
```


```python
!git clone https://github.com/shealyn-wang/ganspace-2
```


```python
%cd ganspace-2
```


```python
model_name = 'StyleGAN2' 
model_class = 'FFHQ-Cartoons' #this is the name of your model in the configs
num_components = 80
```


```python
!pip install fbpca
!pip install boto3
```


```python
!python visualize.py --model $model_name --class $model_class --use_w --layer=style
```

