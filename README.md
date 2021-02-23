# Semi-Supervised Aggregation of Dependent Weak Supervision Sources with Performance Guarantees - Code

## Getting Started

These instructions will setup our experiments to run on your local machine. Since our experiments involve training a large (85!) number of neural networks, it is significantly more efficient to train these in parallel.

### Installing 

In a virtual environment, please install the dependencies using 

```
pip install -r requirements.txt
```
Or alternatively
```
conda install --file requirements.txt
```

You will also need the BatsResearch/labelmodels/ Github repository, which you can find here: https://github.com/BatsResearch/labelmodels. We mainly use this repository for the Semi-Supervised Dawid Skene implementation.

You should clone the repository in the "models/" directory, making "models/labelmodels". After installing the respository, you will need to switch to the semi branch of labelmodels by running 
```
git checkout semi
```

from the "models/labelmodels" directory.

## Setup

Our experiments are performed on the Animals with Attributes 2 dataset. To setup our experiments, you will first need to download the dataset of images and move the Animals with_Attributes folder into the "data/" directory.

Next, you will need to generate our weak supervision sources and convert the image data into numpy matrices, which can be done by running 

```
python setup.py --train True
```

This will first generate numpy files and a pickle file in the "data/" directory of the Animals with Attributes 2 data. Additionaly, this script will train all of the 85 weak labelers. You can batch this up into multiple jobs to significantly speed up training. This script will fine tune pytorch ResNet18s and save the weights in the "data/weak_labelers/" directory.

Next, we will apply each of these trained weak labelers to the image data and save the outputs. You will need to run

```
python setup.py --create True --create_signals True
```

This will create numpy matrices of the weak labelers' soft and hard votes on the images and save them in "data/votes" (hard) and "data/signals" (soft).

## Running Experiments

After creating the weak labelers and storing their hard and soft votes, you can start running our experiments. You can run the experiments that make up the table in our paper by running 

```
python main.py 
```

You can run the experiments used in our figures (with varying amounts of labeled data) by running

```
python vary_labeled_data.py 
```

Both of these scripts have various flags that you can pass in to run the variants of our method or the baselines.

## Citation

Please cite the following paper if you use our work. Thanks!

Alessio Mazzetto, Dylan Sam, Andrew Park, Eliezer Upfal, Stephen H. Bach. "Semi-Supervised Aggregation of Dependent Weak Supervision Sources with Performance Guarantees". Artificial Intelligence and Statistics (AISTATS), 2021.

```
@inproceedings{pgmv,
  title = {Semi-Supervised Aggregation of Dependent Weak Supervision Sources with Performance Guarantees}, 
  author = {Mazzetto, Alessio and Sam, Dylan and Park, Andrew and Upfal, Eliezer, and Bach, Stephen H.}, 
  booktitle = {Artificial Intelligence and Statistics (AISTATS)}, 
  year = 2021, 
}
```