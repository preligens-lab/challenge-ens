"""
Train the model.
"""
from pathlib import Path
import datetime
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from framework.dataset import LandCoverData as LCD
from framework.dataset import parse_image, load_image_train, load_image_test
from framework.model import UNet
from framework.tensorflow_utils import plot_predictions
from framework.utils import YamlNamespace

class PlotCallback(tf.keras.callbacks.Callback):
    """A callback used to display sample predictions during training."""
    from IPython.display import clear_output

    def __init__(self, dataset: tf.data.Dataset=None,
                 sample_batch: tf.Tensor=None,
                 save_folder: Path=None,
                 num: int=1,
                 ipython_mode: bool=False):
        super(PlotCallback, self).__init__()
        self.dataset = dataset
        self.sample_batch = sample_batch
        self.save_folder = save_folder
        self.num = num
        self.ipython_mode = ipython_mode

    def on_epoch_begin(self, epoch, logs=None):
        if self.ipython_mode:
            self.clear_output(wait=True)
        if self.save_folder:
            save_filepaths = [self.save_folder/f'plot_{n}_epoch{epoch}.png' for n in range(1, self.num+1)]
        else:
            save_filepaths = None
        plot_predictions(self.model, self.dataset, self.sample_batch, num=self.num, save_filepaths=save_filepaths)


def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config = config.training_config
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()
    if config.val_samples_csv is not None:
        config.val_samples_csv = Path(config.val_samples_csv).expanduser()
        assert config.val_samples_csv.is_file()

    return config

if __name__ == '__main__':

    import multiprocessing

    config = _parse_args()
    print(f'Config:\n{config}')
    # set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    N_CPUS = multiprocessing.cpu_count()

    print('Instanciate train and validation datasets')
    train_files = list(config.dataset_folder.glob('train/imgs/*.tif'))
    # shuffle list of training samples files
    train_files = random.sample(train_files, len(train_files))
    devset_size = len(train_files)
    # validation set
    if config.val_samples_csv is not None:
        # read the validation samples
        val_samples_s = pd.read_csv(config.val_samples_csv, squeeze=True)
        val_files = [config.dataset_folder/'train/images/{}.tif'.format(i) for i in val_samples_s]
        train_files = [f for f in train_files if f not in set(val_files)]
        valset_size = len(val_files)
        trainset_size = len(train_files)
        assert valset_size + trainset_size == devset_size
    else:
        # generate a hold-out validation set from the training set
        valset_size = int(len(train_files) * 0.1)
        train_files, val_files = train_files[valset_size:], train_files[:valset_size]
        trainset_size = len(train_files) - valset_size

    train_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, train_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)
    val_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, val_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
        .shuffle(buffer_size=1024, seed=config.seed)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    (xp_dir/'tensorboard').mkdir(parents=True)
    (xp_dir/'plots').mkdir()
    (xp_dir/'checkpoints').mkdir()
    # save the validation samples to a CSV
    val_samples_s = pd.Series([int(f.stem) for f in val_files], name='sample_id', dtype='uint32')
    val_samples_s.to_csv(xp_dir/'val_samples.csv', index=False)

    assert False

    # keep a training minibatch for visualization
    for image, mask in train_dataset.take(1):
        sample_batch = (image[:5, ...], mask[:5, ...])

    callbacks = [
        PlotCallback(sample_batch=sample_batch, save_folder=xp_dir/'plots', num=5),
        tf.keras.callbacks.TensorBoard(
            log_dir=xp_dir/'tensorboard',
            update_freq='epoch'
        ),
        # tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=xp_dir/'checkpoints/epoch{epoch}', save_best_only=False, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=(xp_dir/'fit_logs.csv')
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=20,
            factor=0.5,
            verbose=1,
        )
    ]
    # create the U-Net model to train
    unet_kwargs = dict(
        input_shape=(LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
        num_classes=LCD.N_CLASSES,
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print(model.summary())

    # get optimizer, loss, and compile model for training
    optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    # compute class weights for the loss: inverse-frequency balanced
    # note: we set to 0 the weights for the classes "no_data"(0) and "clouds"(1) to ignore these
    class_weight = (1 / LCD.TRAIN_CLASS_COUNTS[2:])* LCD.TRAIN_CLASS_COUNTS[2:].sum() / (LCD.N_CLASSES-2)
    class_weight[LCD.IGNORED_CLASSES_IDX] = 0.
    print(f"Will use class weights: {class_weight}")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    print("Compile model")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
                  # metrics = [tf.keras.metrics.Precision(),
                  #            tf.keras.metrics.Recall(),
                  #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics

    # Launch training
    model_history = model.fit(train_dataset, epochs=config.epochs,
                              callbacks=callbacks,
                              steps_per_epoch=trainset_size // config.batch_size,
                              validation_data=val_dataset,
                              validation_steps=valset_size // config.batch_size,
                              class_weight=class_weight
                              )

