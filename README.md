# TomatoScanner: phenotyping tomato fruit based on only RGB image

**Xiaobei Zhao, Xiangrong Zeng, Yihang Ma, Pengjin Tang, Xiang Li<sup>*</sup>** <br> 📮 first author: 15801175735@163.com <br> 📮 corresponding author: cqlixiang@cau.edu.cn <br> **China Agricultural University** <br><br> Preprint on arXiv: <br> https://arxiv.org/abs/2503.05568

## What TomatoScanner can do
A simple demonstration of TomatoScanner: (a) is the input RGB image. (b) and (c) are the output phenotyping results - width, height, vertical area and volume - of two fruits, respectively. (Zoom in for better observation)
<br> ![TomatoScanner function demonstration](/for_readme/demo.jpg)

## Why we need TomatoScanner
In tomato greenhouse, phenotypic measurement is meaningful for researchers and farmers to monitor crop growth, thereby precisely control environmental conditions in time, leading to better quality and higher yield. Traditional phenotyping mainly relies on manual measurement, which is accurate but inefficient, more importantly, endangering the health and safety of people. Several studies have explored computer vision-based methods to replace manual phenotyping. However, the 2D-based need extra calibration, or cause destruction to fruit, or can only measure limited and meaningless traits. The 3D-based need extra depth camera, which is expensive and unacceptable for most farmers. 

## How TomatoScanner works
In this paper, we propose a non-contact tomato fruit phenotyping method, titled TomatoScanner, where RGB image is all you need for input. First, pixel feature is extracted by instance segmentation of our proposed EdgeYOLO with preprocessing of individual separation and pose correction. Second, depth feature is extracted by depth estimation of Depth Pro. Third, pixel and depth feature are fused to output phenotype results in reality.
<br> ![TomatoScanner architecture](/for_readme/Fig2_TomatoScanner_architecture.jpg)

## How TomatoScanner performs
We establish self-built Tomato Phenotype Dataset to test TomatoScanner, which achieves excellent phenotyping on width, height, vertical area and volume, with median relative error of 5.63%, 7.03%, -0.64% and 37.06%, respectively. We propose and add three innovative modules - EdgeAttention, EdgeLoss and EdgeBoost - into EdgeYOLO, to enhance the segmentation accuracy on edge portion. Precision and mean Edge Error greatly improve from 0.943 and 5.641% to 0.986 and 2.963%, respectively. Meanwhile, EdgeYOLO keeps lightweight and efficient, with 48.7 M weights size and 76.34 FPS.
<br> ![TomatoScanner architecture](/for_readme/Test_experiment_results.png)

<!-- <img src="/for_readme/demo.png" width="70%"> -->


## Set up
Code of TomatoScanner will be open-sourced after publication. 

## Dataset
Dataset of TomatoScanner will be open-sourced after publication.