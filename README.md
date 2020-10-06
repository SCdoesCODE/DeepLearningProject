# Multi-label Classification of X-ray Images

This repository contains the code written for our final project in the Deep Learning course at KTH 

## Abstract

Respiratory diseases often need to be diagnosed by radiologists with the required skill to detect abnormalities in X-ray images. Unfortunately, this is not always readily available in all hospitals and different radiologists can diagnose differently. This project investigated the performance of deep convolutional neural networks with different architectures, both non-pre-trained as well as pre-trained, on the NIH dataset containing 112 000 X-ray images from over 30 000 patients. The results show that the pre-trained VGG16 network with a custom loss function aiming to maximize the f1-score with weighted data according to class-distribution showed the most satisfactory results in terms of accuracy, loss, auc, precision and recall. 

## Methods

### Software

All CNN models were developed using Tensorflow and Keras. All code was written in Python 3. 

### Hardware

All experiments were performed on 1 NVIDIA Tesla T4 GPU through the Google Cloud Platform (GCP) with the machine type being n1-standard-16 (16 vCPUs, 60 GB memory). 
