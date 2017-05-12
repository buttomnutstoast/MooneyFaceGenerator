# MooneyFaceGenerator
A Deep Learned Model for Generating Mooney Faces from Face Dataset

<img src="imgs/pipeline.png" width="250px"/>      <img src="imgs/facescrub_gray.png" width="270px"/>      <img src="imgs/facescrub_bw.png" width="270px"/>

Mooney Face Classification and Prediction by Learning across Tone

[Tsung-Wei Ke](https://www1.icsi.berkeley.edu/~twke/), [Stella X. Yu](https://www1.icsi.berkeley.edu/~stellayu/), [David Whitney](https://whitneylab.berkeley.edu/david_whitney.html)

Mooney faces are special two-tone image, and researchers believe that these images might contain essential element of facial structure which helps huma to percept faces. However, researcher are also bothered with two issues: 1) only small number of Mooney faces available, 2) source photos of these Mooney images are lost.

To address these issues, we propose two models:

1. **Mooney faces generator**
2. **Binary-to-Grayscale images predictor**

We provide source code of **Mooney faces generator** in this repository, and we train a [Pix2Pix GAN](https://phillipi.github.io/pix2pix/) as **Binray-to-Grayscale images predictor**.

## Mooney faces generator

### Prerequisites
* Linux
* NVIDIA GPU + CUDA +CUDNN

### Getting Started
* Install [torch and dependencies](https://github.com/torch/distro)
* Install torch packages `cudnn`, `dpnn`

```
> luarocks install cudnn
> luarocks install dpnn
```

* Clone this repo
```
> git clone https://github.com/buttomnutstoast/MooneyFaceGenerator.git
> cd MooneyFaceGenerator
```

* Prepare the datasets
```
> Need to be finished
```

### Train the Mooney face classifier
We first fine-tune the facial recognition model by [openface](https://cmusatyalab.github.io/openface/) for grayscale face classification. You can download the model from [here](#) and then fine-tune it to mooney face classifier.
```
> DATA_ROOT=/path/to/root/dir/of/datasets TRAINED_MODEL_PATH=/path/to/trained/models sh scripts/mooney_train.sh
```

### Generate Mooney faces
```
> DATA_ROOT=/path/to/root/dir/of/datasets TRAINED_MODEL_PATH=/path/to/trained/models sh scripts/mooney_train.sh
```

### Filter most-likely Mooney candidates from each images
```
> Need to be finished
```

## Binary-to-Grayscale images predictor

### Setup
Please follow the instruction and download Pix2Pix GAN from [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix).

### Dataset prepration
You can,

1. Use the faces generated from 
2. Download our dataset from [here](#)

---

## Acknowledgements
We borrow code heavily from [Southmith](https://github.com/soumith/imagenet-multiGPU.torch). Also, we fine-tune the model `nn4small2v1` trained by [openface](https://cmusatyalab.github.io/openface/) to mooney face classifier.
