# CaricatureFace
This repository includes source code, pretrained model and a testset of paper "Landmark Detection and 3D Face Reconstruction for Caricature using a Nonlinear Parametric Model", [http://arxiv.org/abs/2004.09190](http://arxiv.org/abs/2004.09190).

![examples](fig/examples.png)

## Landmarks vertex indexes

68 landmark indexes of 6144 vertexes:

```shell script
1316, 822, 3920, 3842, 3879, 1083,  504, 1080, 3611, 2643, 1929, 2646,
5334, 5363, 5292, 2247, 2741,  702,  711,  853,  724,  856, 4875, 4848,
2150, 2374, 2386, 2394, 3901, 2914, 2890, 3549,  355, 2931, 1780, 5686,
137,  619, 4340, 4401,  154,  150, 4744, 2340, 5072, 2335, 2344, 4814,
3196, 3237, 3254, 6072, 6142, 6124, 6049, 6045, 1617, 3190,  192, 3267,
256, 1459, 3066, 3065, 2953, 6014, 3069, 3234
```

## Update Logs:
### May 9, 2020
* Add a toy example for conversion between 3D face and vertices.
### April 22, 2020
* The testset is enlarged.
### April 8, 2020
* The source code, pretrained model and some data from testset are released.


## Comparison with us
If you want to do some comparison with our method, you can download a testset here [Google Drive](https://drive.google.com/open?id=1fGHlV8ISUkgCK8OSTQxvEJxtxXXrwjDI), or [Baidu Drive](https://pan.baidu.com/s/1YhniT8yb6C5yvO9gq_YYoA) with password: 4nvs. It includes 2D caricatures, groundtruth 68 landmarks, 68 landmarks detected by our method and 3D meshes recovered by our method.

## Prerequisites and Installation
- Python 3.7
- Pytorch 1.4.0
- opencv-python 3.4.2

### Getting Started
**Clone this repository:**
```bash
git clone git@github.com:Juyong/CaricatureFace.git
cd CaricatureFace
```
**Install dependencies using Anaconda:**
 ```bash
conda create -n cariface python=3.7
source activate cariface
pip install -r requirements.txt
```

## Advanced Work
**Prepare related data:**
- You can download related data for alogorithm here [Google Drive](https://drive.google.com/open?id=11m9dC6j-SUyjhtSiXsUqiBdZOQ3S8phD), or [Baidu Drive](https://pan.baidu.com/s/1v4V-7rYszDhyhzhCH2aYeA) with password: tjps.
- Unzip downloaded files and move files into ```./data``` directory.

**Prepare pretrained model:**
- You can download pretrained model here [Google Drive](https://drive.google.com/open?id=1If_rjQp5mDZMbK1-STGYOPyw_cTG66jO), or [Baidu Drive](https://pan.baidu.com/s/113QFM-zhSUIZfzjFhQfTTA) with password: fukf.
- Unzip downloaded files and move files into ```./model``` directory.

**Prepare some examples:**
- You can download some examples here [Google Drive](https://drive.google.com/open?id=1X8TpVpGzRrQuSS93_Hb32ERU-P4q6SSG), or [Baidu Drive](https://pan.baidu.com/s/18V7Ulq9ZfACtuhpkBJY6-A) with password: unud.
- Unzip downloaded files and move files into ```./exp``` directory.

## Test with Pretrained Model
Within ```./CaricatureFace``` directory, run following command:
 ```bash
    bash test.sh
```

Note: Input images must be preprocessed - crop the whole face roughly and resize to size (224, 224).

## Recover 3D faces
Please follow README.txt in ```./toy_example``` directory.

## Citation
If you find this useful for your research, please cite the paper:
```
@article{Zhang2020Caricature,
  author    = {Juyong Zhang and
               Hongrui Cai and
               Yudong Guo and
               Zhuang Peng},
  title     = {Landmark Detection and 3D Face Reconstruction for Caricature using a Nonlinear Parametric Model},
  journal   = {CoRR},
  volume    = {abs/2004.09190},
  year      = {2020}
}
``` 
