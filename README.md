# ENS Challenge Data 2021 - S2GLC



Welcome to this truly not awesome repository !

### Data

The data is stored on AWS S3 at the URI: `s3://earthcube-projects/1-customer_project/mmt_multispectral/Challenge_ENS/v1_dataset`. The file structure of the directory is:

```
v1_dataset
├── test
│   ├── imgs [5043 entry files of form <country>_x_y.tif]
│   └── masks [5043 entry files of form <country>_x_y.tif]
├── test_images.csv
├── test_images_name2id.csv
├── test_labels.csv
├── test_labels_predicted_malicious.csv
├── test_labels_predicted_random.csv
├── train
│   ├── imgs [18491 entry files of form <country>_x_y.tif]
│   └── masks [18491 entry files of form <country>_x_y.tif]
├── train_images.csv
├── train_images_name2id.csv
└── train_labels.csv
```

The images are 16-bits GeoTIFF files of size (256,256,4) and the masks are 8-bits GeoTIFF files of size (256,256)

Every sample has an identifier computed in increasing order of file name, the identifiers are used in the CSVs in the column named `sample_id`. The CSVs `<DDDD>_image_name2id.csv` with`<DDDD>` equal to `train` or `test` map the images paths composing the training or testing set respectively to their unique identifier in the dataset.

### Environment

The file `environment.yaml` is an exported conda environment with all the dependencies for this project: you can recreate the environment with the command `conda env create -f environment.yaml`. You first need to install miniconda if you don't have it already: follow [these steps](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) for installation on Linux.
