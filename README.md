donnot finish
## Weakly Supervised Image Retrieval via Coarse-scale Feature Fusion and Multi-level Attention Blocks

This repository is the code for [*Weakly Supervised Image Retrieval via Coarse-scale Feature Fusion and Multi-level Attention Blocks*](http://delivery.acm.org/10.1145/3330000/3325017/p48-nie.pdf?ip=202.120.235.148&id=3325017&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E88014DC677A1F2C3%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1568898900_445f14daa72cc4ac91920063cece9a18) in ICMR 2019.

![network](https://github.com/SHIsue/ABIR/blob/master/images/network.png)

### Requirements

Python 3, PyTorch >= 0.4.0, and make sure you have installed TensorboardX:

```
pip install tensorboardX
```

### Quick Start

__1\. Prepare the Dataset__

Download four datasets: [In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html), [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Online Products](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16) and [Cars-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 


### Results

We conduct the experiments on all commonly adopted image retrieval task datasets and utilize Recall@K metric for evaluation. 

The following table shows the results on the In-Shop Clothes Retrieval dataset. Best results are marked in bold.

|R@           | 1          | 10         | 20        | 30        | 40        | 50        |
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| [FashionNet+Joints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)  | 41.0     | 64.0     | 68.0   | 71.0   | 73.0      | 73.5      |
| [FashionNet+Poselets](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)         | 42.0     | 65.0     | 70.0   | 72.0   | 75.0      |
| [FashionNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780493)        | 53.0     | 73.0     | 76.0   | 77.0   | 79.0      | 80.0      |
| [HDC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237356) | 62.1     | 84.9     | 89.0   | 91.2   | 92.3      | 93.1      |
| [HTL](https://arxiv.org/pdf/1810.06951.pdf)        | 80.9 | 94.3 | 95.8   | 97.2   | 97.4      | 97.8  | 
| [A-BIER](https://arxiv.org/pdf/1810.06951.pdf)        | 83.1 | 95.1 | 96.9   | 97.5   | 97.8      | 98.0  | 
| [ABE-8](https://arxiv.org/pdf/1810.06951.pdf)        | 87.3 | 96.7 | 97.9   | 98.2   | 98.5      | 98.7  | 
|-------------|------------|------------|-----------|-----------|-----------|-----------|
| Our Baseline        | 85.4 | 96.1 | 97.3   | 97.8   | 98.1      | 98.3  | 
| ABIR w/o SE-block        | 88.1 | 96.9 | 97.6   | 98.1   | 98.3      | 98.5  | 
| ABIR with SE-block        | **89.0** | **97.1** | **98.0**   | **98.4**   | **98.6**      | **98.8**  | 

The following table shows the results on the In-Shop Clothes Retrieval dataset. Best results are marked in bold.

