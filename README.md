# SimVAE

Official PyTorch codebase for SimVAE presented in **A Probabilistic Model Behind Self-Supervised Representation Learning** 
[\[arXiv\]](https://arxiv.org/pdf/2402.01399).

## Method 
SimVAE is trained by maximising $ELBO_{SSL}$, the evidence lower bound to the graphical model depicted below. 
$ELBO_{SSL}$ considers a mixture of $N$ Gaussians as a prior with $N$ corresponding to the number of training samples. 
Maximizing $ELBO_{SSL}$ promotes high likelihood for a sample $x_i$ and its augmentations {$x_i^j$}$_j$ under the $i$ -th Gaussian while minimizing the reconstruction error.

<p align="center">
    <img src="https://github.com/alicebizeul/simvae/blob/main/assets/graphical_model.png" alt="simvae" width="200">
</p>


## Code Structure

```
.
├── assets                       # assets for the README file 
├── scripts                      # bash scripts to launch training and evaluation
│   ├── train.sh                 #   training script
│   └── eval.sh                  #   evaluation script
├── src                          # the package
│   ├── utils                    #   utilities
│   │   ├── plotting.py          #     plotting functions
│   │   ├── options.py           #     arguments
│   │   └── utils.py             #     utilities functions
│   ├── model                    #   model 
│   │   ├── models.py            #     models
│   │   ├── decoders.py          #     decoder architecture
│   │   ├── resnet.py            #     encoder architecture
│   │   └── priors.py            #     priors
│   └── data                     #   data
│       └── data_loader.py       #     data loader
├── main.py                      # entrypoint to launch SimVAE training
└── requirements.txt             # requirements file
```

## Installation 

In your environment of choice, install the necessary requirements

    !pip install -r requirements.txt 

Alternatively, install individual packages as follows:

    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install pandas numpy pillow scikit-learn scikit-image matplotlib tqdm

## Launch Training 
To launch experiments, you can find training and evaluation scripts in ```scripts```. The following modifications should be made to the ```train.sh``` script to ensure a smooth training on your local machine:

    DATA_PATH=mypath2data
    OUTPUT_PATH=mypath2outputs

The ```mypath2data``` folder should point to the directory which contains either the ```cifar-10-batches-py``` folder for CIFAR10, the ```MyMNIST``` folder containing ```raw``` for MNIST, the ```MyFashionMNIST``` folder containing ```raw``` for FashionMNIST. The ```mypath2outputs``` should point towards a directory where the run ouputs will be saved. You can easily change the dataset on which to run SimVAE by adjusting the ```DATASET``` environment variable in ```train.sh``` to either ```cifar10```, ```mnist``` or ```fashionmnist```.
An overview of all hyperparameters that can be modulated can be found in ```/src/utils/options.py```. 

---
