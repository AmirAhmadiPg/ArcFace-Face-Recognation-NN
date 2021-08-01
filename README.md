# Welcome to the ArcFace Face Recognation NN wiki!
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [output example](#output-example)

# General info
ArcFace is a network for Face Recognation or any problem that need to classification in discriminative space
like: retina id, fingerprint, and some more...

the only special thing for coding in arc face is to define **ArcFace Loss function.**
**You can find it in arcface_layer file**

### Normilization:
Data Normalization is: pic/255
Labels: One Hot encoding


## Technologies
Project is created with:

* language: **python**
* python version: **3.8.8**
* libs: **Tensorflow, keras**

## Setup
To run this project, install it locally using python:
##### Its better if we use GPU(CUDA) to run this code
```
$ cd /addres/to/project/folder
$ python3 Mymodel.py
```

## output example
### output (Mnist Dataset)
![alt text](./Arcade_output.jpg?raw=true)

