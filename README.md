# Image completer for MNIST
GAN based MNIST image completer.

Project built based off of the paper - http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf and the repo - https://github.com/otenim/GLCIC-PyTorch

## Installing environment
Install Anaconda or miniconda and run 

```
conda env create -f environment.yml
```

## To download dataset
Please run `create_mnist.py` in the datasets folder to download and create the mnist data folders

## Running train script
After downloading the dataset, please run :

``` 
python train.py datasets/mnist/ results/
```

The results folder is created with models and results of the Phase 1, Phase 2 and Phase 3 of training


## Running test script
Assuming results are written to a directory `results_1`, please run -
```
python predict.py ./results_1/phase_3/model_cn_step10000 ./results_1/config.json ./datasets/mnist/test/341.jpeg ./result_images/test_run_2.jpeg
```

There are few results already available in the folder `result_images`