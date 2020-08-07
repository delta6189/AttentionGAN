# AttentionGAN

Unofficial implementation of AttentionGAN with pytorch

Official implementation: https://github.com/Ha0Tang/AttentionGAN

Prerequisites
------

  `pytorch`
  
  `torchvision`
  
  `numpy`
  
  `openCV2`
  
  `matplotlib`
    
Dataset
------

  https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
    
Train
------

  Please refer `train.ipynb`
  
Test
------

  Please refer `test.ipynb`
  
  
Training details
------

| <center>Parameter</center> | <center>Value</center> |
|:--------|:--------:|
| Learning rate | 2e-4 | 
| Batch size | 1 | 
| Epoch | 100 | 
| Optimizer | Adam |
| (beta1, beta2) | (0.5, 0.999) |
| (gamma, lambda1, lambda2) | (10, 10, 1) |
| Data Augmentation | RandomHorizontalFlip() |
| HW | CPU : Intel i5-8400<br>RAM : 16G<br>GPU : NVIDIA GTX1060 6G |
| Training Time | About 20 hours for 100 epoch |

Model
------

Please refer original paper (https://arxiv.org/abs/1903.12296)
 

 
Results
-----

<center>Summer <-> Winter</center>  

![ex_screenshot](./sample/sample1.png)
![ex_screenshot](./sample/sample2.png)
