# Zooming-Slow-Mo (CVPR-2020)
This is the official Pytorch implementation of *Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution*.

---

*This repo is currently under construction. The complete version of code will be released by June. 14, 2020*

By [Xiaoyu Xiang<sup>*</sup>](https://engineering.purdue.edu/people/xiaoyu.xiang.1), [Yapeng Tian<sup>*</sup>](http://yapengtian.org/), [Yulun Zhang](http://yulunzhang.com/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/), [Jan P. Allebach<sup>+</sup>](https://engineering.purdue.edu/~allebach/), [Chenliang Xu<sup>+</sup>](https://www.cs.rochester.edu/~cxu22/) (<sup>*</sup> equal contributions, <sup>+</sup> equal advising)

#### [Paper](https://arxiv.org/abs/2002.11616) | [Project Page] | [Demo Video](https://youtu.be/8mgD8JxBOus) 

[![Watch the video](dump/demo_thumbnail.PNG)](https://youtu.be/8mgD8JxBOus)

## Prerequisites

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). 
- Python packages: `pip install numpy opencv-python lmdb pyyaml`

## Pretrained Models

Our pretrained model can be downloaded via [GitHub](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/experiments/pretrained_models/xiang2020zooming.pth) or [Google Drive](https://drive.google.com/open?id=1xeOoZclGeSI1urY6mVCcApfCqOPgxMBK).

## Get Started
### Installation 

### Training


### Testing
We provide the test code for both standard test sets (Vid4, SPMC, etc.) and custom images. 

You can change the input path, number of frames, etc. in [test.py](codes/test.py).

- Test the model:
```bash
cd codes
python test.py
```

- The test results will be saved to a folder here: `./results/your_data_name/`.


## Citations
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{xiang2020zooming,
  author = {Xiang, Xiaoyu and Tian, Yapeng and Zhang, Yulun and Fu, Yun and Jan, Allebach and Xu, Chenliang},
  title = {Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time VideoSuper-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}

@InProceedings{tian2018tdan,
  author={Yapeng Tian, Yulun Zhang, Yun Fu, and Chenliang Xu},
  title={TDAN: Temporally Deformable Alignment Network for Video Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}

@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

## License
This project is released under the [GNU General Public License v3.0](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/LICENSE).

## Acknowledgments
Our code is inspired by [TDAN-VSR](https://github.com/YapengTian/TDAN-VSR) and [EDVR](https://github.com/xinntao/EDVR).