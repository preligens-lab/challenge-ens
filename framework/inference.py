"""
Predict the targets using a trained model.
"""
from pathlib import Path
import argparse
import yaml
from tifffile import TiffFile
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from framework.dataset import LandCoverData as LCD
from framework.model import UNet
from framework.utils import YamlNamespace

# random seed for reproducibility
SEED = 42

def numpy_parse_image(image_path):
    """Load an image as numpy array
    Args:
        image_path (bytes): path to image
    Returns:
        numpy.array[uint8]: the image array
    """
    image_path = Path(bytes.decode(image_path))
    with TiffFile(image_path) as tifi:
        image = tifi.asarray()
    return image

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def parse_image(image_path):
    """Wraps the parse_image function as a TF function"""
    image, = tf.numpy_function(numpy_parse_image, (image_path,), (tf.uint16,))
    image.set_shape([LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS])
    return image

@tf.function
def normalize(input_image):
    """Rescale the pixel values of the images between 0.0 and 1.0"""
    image = tf.cast(input_image, tf.float32) / LCD.TRAIN_PIXELS_MAX
    return image

@tf.function
def load_image_test(input_image):
    """Normalize test image"""
    image = normalize(input_image)
    return image


def predict_as_vectors(model, dataset, save_to=None):
    """Perform a forward pass over the dataset and bincount the prediction masks to return class vectors.
    Args:
        model (tf.keras.Model): model
        test_dataset: (tf.data.Dataset): dataset to perform inference on
    Returns:
        (pandas.DataFrame): predicted class distribution vectors for the dataset
    """
    def bincount_along_axis(arr, minlength=None, axis=-1):
        """Bincounts a tensor along an axis"""
        if minlength is None:
            minlength = tf.reduce_max(arr) + 1
        mask = tf.equal(arr[..., None], tf.range(minlength, dtype=arr.dtype))
        return tf.math.count_nonzero(mask, axis=axis-1 if axis < 0 else axis)

    predictions = []
    for batch in tqdm(dataset):
        # predict a raster for each sample in the batch
        pred_raster = model.predict_on_batch(batch)

        (batch_size, _, _, num_classes) = tuple(pred_raster.shape)
        pred_mask = tf.argmax(pred_raster, -1) # (bs, 256, 256)
        # bincount for each sample
        counts = bincount_along_axis(
            tf.reshape(pred_mask, (batch_size, -1)), minlength=num_classes, axis=-1
        )
        predictions.append(counts / tf.math.reduce_sum(counts, -1, keepdims=True))

    predictions = tf.concat(predictions, 0)
    return predictions.numpy()


def _parse_args():
    parser = argparse.ArgumentParser('Inference script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    parser.add_argument('--xp-dir', '-x', type=str, required=True,
                        help="The path to the log directory for an experiment."
                             "If 'last' will use the lastly created directory under config.xp_rootdir"
                        )

    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config = config.inference_config
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()
    # special args
    if cli_args.xp_dir == 'last':
        # get last xp directory name
        cli_args.xp_dir = Path(max(str(d) for d in config.xp_rootdir.iterdir() if d.is_dir()))
    else:
        config.xp_dir = config.xp_rootdir/cli_args.xp_dir
    assert config.xp_dir.is_dir()
    return config


if __name__ == '__main__':

    import multiprocessing

    config = _parse_args()
    N_CPUS = multiprocessing.cpu_count()

    print('Instanciate test dataset')
    test_filenames = sorted(config.dataset_folder.glob('test/imgs/*.tif'))
    test_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, test_filenames)))
    # check samples are loaded in the right order
    # for idx, (filename, tensor) in enumerate(zip(test_filenames, test_dataset)):
    #     try:
    #         assert filename == str(bytes.decode(tensor.numpy()))
    #     except AssertionError:
    #         print(filename)
    #         print(str(bytes.decode(tensor.numpy())))
    #         raise

    test_dataset = test_dataset.map(parse_image, num_parallel_calls=N_CPUS)\
        .map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat(1)\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Load the trained model saved to disk
    try:
        model = tf.keras.models.load_model(str(config.xp_dir/f'checkpoints/epoch{config.checkpoint_epoch}'))
    except:
        raise

    print("Predict the vectors over the test dataset")
    y_pred_test = predict_as_vectors(model, test_dataset)
    df_y_true_test = pd.read_csv(config.dataset_folder/'new_csvs/test_labels.csv', index_col=0)
    df_y_pred_test = pd.DataFrame(y_pred_test, index=df_y_true_test.index, columns=df_y_true_test.columns)
    out_csv = config.xp_dir/f'epoch{config.checkpoint_epoch}_test_predicted.csv'
    print(f"Saving prediction CSV to file {str(out_csv)}")
    df_y_pred_test.to_csv(out_csv, index=True, index_label='sample_id')

    print(df_y_pred_test.shape, df_y_pred_test.values.dtype)
    print(df_y_true_test.shape, df_y_true_test.values.dtype)
