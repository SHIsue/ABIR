donnot finish
## Weakly Supervised Image Retrieval via Coarse-scale Feature Fusion and Multi-level Attention Blocks

Xinyao Nie, Hong Lu, Zijian Wang, Jingyuan Liu, Zehua Guo

Fudan University

This repository is the code for [*Weakly Supervised Image Retrieval via Coarse-scale Feature Fusion and Multi-level Attention Blocks*](http://delivery.acm.org/10.1145/3330000/3325017/p48-nie.pdf?ip=202.120.235.148&id=3325017&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1569220109_4d00dffbd4d9fffc5306e5d1173efed8) in ICMR 2019. If you cannot open the link, you can just download the paper which is uploaded in pdf format.

![network](https://github.com/SHIsue/ABIR/blob/master/images/network.png)

### Requirements

Python 3, PyTorch >= 0.4.0, and make sure you have installed TensorboardX:

```
pip install tensorboardX
```

### Quick Start

__1\. Prepare the Dataset__

Our work utilized four datasets: [In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Online Products](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16) and [Cars-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html). 
In this repo, we just focus on [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset. You can use "get_cub_train_file.py" to generate "train.txt" and "test.txt" OR download "train.txt" and "test.txt" directly.

__2\. Repo Structure__

```text
  ABIR
      |———— data/                                 # store CUB-200-2011 dataset
          |———— cub/
                  |———— images/
                  |———— images.txt                # you can use image.txt, image_class_labels.txt and train_test_split.txt to 
                  |———— image_class_labels.txt    # generate train.txt and test.txt
                  |———— train_test_split.txt
                  |———— train.txt
                  |———— test.txt
      |———— code/                               
          |———— models/              
                  |———— VGG16_V5.py
          |———— __init__.py
          |———— config.py                               
          |———— Model2Feature.py 
          |———— train.py  
          |———— test.py  
          |———— trainer.py  

 
```


__3\. Train the Model__

Run "train.py".


### Results

We conduct the experiments on all commonly adopted image retrieval task datasets and utilize Recall@K metric for evaluation. 

The following table shows the results on the In-Shop Clothes Retrieval dataset. Best results are marked in bold.

|R@           | 1          | 10         | 20        | 30        | 40        | 50        |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| [FashionNet+Joints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)  | 41.0     | 64.0     | 68.0   | 71.0   | 73.0      | 73.5      |
| [FashionNet+Poselets](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)         | 42.0     | 65.0     | 70.0   | | 72.0     |72.0     | 75.0      |
| [FashionNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)        | 53.0     | 73.0     | 76.0   | 77.0   | 79.0      | 80.0      |
| [HDC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237356) | 62.1     | 84.9     | 89.0   | 91.2   | 92.3      | 93.1      |
| [HTL](https://arxiv.org/pdf/1810.06951.pdf)        | 80.9 | 94.3 | 95.8   | 97.2   | 97.4      | 97.8  |
| [A-BIER](https://arxiv.org/pdf/1810.06951.pdf)        | 83.1 | 95.1 | 96.9   | 97.5   | 97.8      | 98.0  |
| [ABE-8](https://arxiv.org/pdf/1810.06951.pdf)        | 87.3 | 96.7 | 97.9   | 98.2   | 98.5      | 98.7  |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| Our Baseline        | 85.4 | 96.1 | 97.3   | 97.8   | 98.1      | 98.3  |
| ABIR w/o SE-block        | 88.1 | 96.9 | 97.6   | 98.1   | 98.3      | 98.5  |
| ABIR with SE-block        | **89.0** | **97.1** | **98.0**   | **98.4**   | **98.6**      | **98.8**  |

The following table shows the results on the CUB-200-2011 dataset. Best results are marked in bold.

|R@           | 1          | 10         | 20        | 30        | 40        | 50        |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| [margin](https://arxiv.org/pdf/1706.07567.pdf)  | 63.9   | 75.3     | 84.4   | 90.6   | 94.8    | -   |
| [HDC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237356)         | 60.7     | 72.4     | 81.9   | 89.2   | 93.7    | 96.8   |
| [HTL](https://arxiv.org/pdf/1810.06951.pdf)        | 57.1     | 68.8     | 78.7   | 86.5   | 92.5    | 95.5      |
| [A-BIER](https://arxiv.org/pdf/1810.06951.pdf) | 65.5     | 75.8     | 83.9   | 90.2   | 94.2      | **97.1**      |
| [ABE-8](https://arxiv.org/pdf/1810.06951.pdf)        | 60.6 | 71.5 | 79.8   | 87.4   | -      | -  |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| Our Baseline        | 73.1 | 81.9 | 87.6   | 91.4   | 93.8   | 96.2  |
| ABIR w/o SE-block        | 77.5 | 84.1 | 88.7   | 91.7   | 94.2      | 96.3  |
| ABIR with SE-block         | **78.1** | **84.6** | **88.7**   | **91.8**   | **94.4**      | 96.6  |

The following table shows the results on the Stanford Online Products dataset. Best results are marked in bold.

|R@           | 1          | 10         | 100        | 1000        |
|-------------|------------|------------|-----------|-----------|
| [Contrastive](https://arxiv.org/pdf/1511.06452.pdf)  | 42.0   | 58.2     | 73.8   | 89.1   |
| [Triplet](https://arxiv.org/pdf/1511.06452.pdf)  | 42.1   | 63.5     | 82.5   | 94.8   |
| [LiftedStruct](https://arxiv.org/pdf/1511.06452.pdf)  | 62.1   | 79.8     | 91.3   | 97.4   |
| [HDC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237356)         | 69.5     | 84.4     | 92.8   |  97.7   |
| [HTL](https://arxiv.org/pdf/1810.06951.pdf)        | 74.8     | 88.3     | 94.8   | 98.4   |
| [A-BIER](https://arxiv.org/pdf/1810.06951.pdf) | 74.2     | 86.9     | 94.0   | 97.8   |
| [ABE-8](https://arxiv.org/pdf/1810.06951.pdf)        | **76.3** | **88.4** | 94.8   | 98.2   |
|-------------|------------|------------|-----------|-----------|
| Our Baseline        | 71.2 | 85.6 | 93.5   | 97.7   |
| ABIR w/o SE-block        | 74.3 | 87.4 | 94.6   | 98.3   |
| ABIR with SE-block         | 74.8 | 87.7 | **95.0**   | **98.5**   |

The following table shows the results on the Cars-196 dataset. Best results are marked in bold.

|R@           | 1          | 2         | 4        | 8        | 16        | 32        |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| [margin](https://arxiv.org/pdf/1706.07567.pdf)  | 86.9   | 92.7     | 95.6   | **97.6**   | **98.7**    | -   |
| [HDC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237356)         | 83.8     | 89.8     | 93.6   | 96.2   | 97.8    | 98.9   |
| [HTL](https://arxiv.org/pdf/1810.06951.pdf)        | 81.4     | 88.0     | 92.7   | 95.7   | 97.4    | **99.0**      |
| [A-BIER](https://arxiv.org/pdf/1810.06951.pdf) | 82.0     | 89.0     | 93.2   | 96.1   | -      | -      |
| [ABE-8](https://arxiv.org/pdf/1810.06951.pdf)        | 85.2 | 90.5 | 94.0   | 96.1   | -      | -  |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| Our Baseline        | 82.6 | 88.1 | 92.4   | 95.3   | 97.4   | 98.4  |
| ABIR w/o SE-block        | 89.1 | 93.1 | 95.4   | 97.2   | 98.3      | 99.1  |
| ABIR with SE-block         | **89.4** | **93.3** | **95.6**   | 97.1   | 98.2      | **99.0**  |

### Citation

If this code helps your research, please cite our paper:

```shell
@inproceedings{nie2019weakly,
  title={Weakly Supervised Image Retrieval via Coarse-scale Feature Fusion and Multi-level Attention Blocks},
  author={Nie, Xinyao and Lu, Hong and Wang, Zijian and Liu, Jingyuan and Guo, Zehua},
  booktitle={Proceedings of the 2019 on International Conference on Multimedia Retrieval},
  pages={48--52},
  year={2019},
  organization={ACM}
}
```

