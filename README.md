# ENS Challenge Data 2021 : Land cover predictive modeling from satellite images

This repository stores the code for the benchmark model of the Challenge Data competition ["Land cover predictive modeling from satellite images"](https://challengedata.ens.fr/challenges/48).
The proposed benchmark model is a deep neural network trained on the “proxy” task of semantic segmentation of the land cover labels at the pixel level. The network has a U-Net architecture ([Ronneberger et al 2015](https://arxiv.org/abs/1505.04597)).

## Data folder

You can download the data as an archive containing the training images and masks, as well as the test images, from the challenge page.

The dataset folder should be like this :
```
dataset_UNZIPPED
├── test
│   └── images
│       ├── 10087.tif
│       ├── 10088.tif
│       ├── 10089.tif
│       ├── 10090.tif
        ... (5043 files)
└── train
    ├── images
    │   ├── 10000.tif
    │   ├── 10001.tif
    │   ├── 10002.tif
    │   ├── 10003.tif
        ... (18491 files)
    └── masks
        ├── 10000.tif
        ├── 10001.tif
        ├── 10002.tif
        ├── 10003.tif
        ... (18491 files)
```
The images are 16-bits GeoTIFF files of size (256,256,4) and the masks are 8-bits GeoTIFF files of size (256,256).

Every sample has an identifier used in the CSVs in a column named `sample_id`.

## Python environment

The file `environment.yaml` is an exported conda environment with all the dependencies for this project: you can recreate the environment with the command `conda env create -f environment.yaml`. You first need to install miniconda if you don't have it installed already: go to [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to find instructions for your operating system.

## Code usage

The notebook `data_visualization.ipynb` can be used to visualize a few data points, as well as the class distribution in the training set. Notably you'll see how to load the images and masks using `tifffile`.

The `framework` module contains scripts to train the model, then use the trained model to perform predictions over the testing set, and finally evaluate those predictions against corresponding ground truth labels.

### Train

To train the U-Net model on the training set for a certain number of epochs, use `framework/train.py` :
```
ipython framework/train.py -- --config config.yaml
```
The input is a YAML configuration file. See `train_config.yaml` for an example file with description of parameters, and values used in the benchmark.

The experiment is saved in a folder with a structure like this :
```
experiments
├── 25-11-2020_20:40:43
│   ├── checkpoints
│   ├── plots
|   ├── fit_logs.csv
|   ├── tensorboard
│   └── val_samples.csv
```
Here, `experiments` is the root folder referenced by `xp_rootdir` in `config.yaml`. In the root folder, a directory for the experiment is created with name being the datetime of execution (here: `25-11-2020_20:40:43`). Inside, `checkpoints` contains snapshots of the model, saved after every epoch. `plots` contains the figures of predictions of the model after every epoch for a few train samples: it helps to qualitatively judge the progress of the training. The file `fit_logs.csv` contains logs of the training with loss and validation loss for every epoch. The file `val_samples.csv` contains the samples kept in validation during the training (the weights of the model are not optimized on those samples).

### Predict

After having trained your model, you can use `framework/infer.py` to predict the class-distribution vector targets over samples in the train, validation, or test set, and saved them in a CSV file, with the right format for submission:
```
ipython framework/infer.py -- --config config.yaml
```
The input is a YAML configuration file. See `infer_config.yaml` for an example file with description of parameters.

### Evaluate

The script `framework/eval.py` does the same thing as the evaluation done for submissions on the server. It computes the metric of the challenge (the Kullback-Leibler divergence) between a prediction CSV and a ground-truth CSV. Usage :
```
ipython framework/eval.py -- --gt-file path/to/labels.csv --pred-file path/to/predicted.csv -o /path/to/save.csv
```
Examples:
* Evaluate on training set:

```
ipython framework/eval.py -- --gt-file ../data/dataset_csvs/train_labels.csv --pred-file experiments/25-11-2020_20:40:43/epoch84_train_predicted.csv --out-csv experiments/25-11-2020_20:40:43/epoch84_train_predicted_score.csv
```
where ../data/dataset_csvs/train_labels.csv is where is stored the *y_train* file of the challenge.

* Evaluate on validation set:
```
ipython framework/eval.py -- --gt-file ../data/dataset_csvs/train_labels.csv --pred-file experiments/25-11-2020_20:40:43/epoch84_val_predicted.csv --out-csv experiments/25-11-2020_20:40:43/epoch84_val_predicted_score.csv
```


## Model information

U-Net is composed of a contractive path and expansive path. The contractive path diminishes the spatial resolution by the repeated application of (3,3) convolutions and (2,2) max pooling with a stride of 2. Every step in the expansive path consists in (2,2) transposed convolutions that expand the spatial resolution, a concatenation with the feature map in correspondence in the contractive path, followed by (3,3) convolutions. At the last layer, that has the same resolution as the input, a (1,1) convolution is used to associate the feature map vector for every pixel to the number of classes.
The benchmark model is a relatively small network made of 984,234 trainable parameters. The total number of convolution layers is 21, made of 64 feature maps in the contractive path, and 64 or 96 feature maps in the expansive path. We kept the number of feature maps low to be able to have this much layers while keeping the number of weights small. See `framework/model.py:UNet` for the exact definition of the model in Keras code.

The loss function used is a regular cross-entropy, with weights assigned to every class. The weight for a class is set to be the inverse of the frequency of this class in the training set. The special classes “no_data” and “clouds” need to be ignored have their weights set to zero to avoid learning to predict them.

As data augmentations during training, we used basic flips and rotations (90°, 270°).
The model was trained for 90 epochs which took about 6 hours on a Nvidia Tesla P100-PCIE-16GB.
To select a snapshot for predicting the class distribution vectors on the testing set, we selected the one which is optimal for validation loss (the 86th epoch).

Note: the learning rate used is 0.001 and it wasn't tuned at all. The batch size is 32.

The architecture used is inspired by [DeepSense.AI Kaggle DSTL competition entry](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/). Refer to the “Model” section of the article and see [their model's figure](https://cdn-sv1.deepsense.ai/wp-content/uploads/2017/04/architecture_details.png), ours is very similar.
