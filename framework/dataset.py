"""
Classes and functions to handle data
"""
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tifffile import TiffFile
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
# import tensorflow_io as tfio

# random seed for reproducibility
SEED = 42

class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge, with useful
    metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES_DICT = OrderedDict({
        0: 'no_data',
        1: 'clouds',
        2: 'artificial',
        3: 'cultivated',
        4: 'broadleaf',
        5: 'coniferous',
        6: 'herbaceous',
        7: 'natural',
        8: 'snow',
        9: 'water',
    })
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES = ['no_data', 'clouds']

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: color palette to use
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training sets
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356

    def __init__(self, dataset_folder):
        dataset_folder = Path(dataset_folder)
        assert dataset_folder.is_dir()
        self.train_images_paths = sorted(list(dataset_folder.glob('train/imgs/*.tif')))
        self.train_masks_paths = sorted(list(dataset_folder.glob('train/masks/*.tif')))
        self.test_images_paths = sorted(list(dataset_folder.glob('test/imgs/*.tif')))
        self.test_masks_paths = sorted(list(dataset_folder.glob('test/masks/*.tif')))
        assert len(train_images_paths) == self.TRAINSET_SIZE
        assert len(test_images_paths) == self.TESTSET_SIZE

    def show_image(self, image, display_min=50, display_max=400, ax=None):
        """Show an image.
        Args:
            image (numpy.array): the image. If the image is 16-bit, apply bytescaling to convert to 8-bit
        """
        if image.dtype == np.uint16:
            iscale = display_max - display_min
            scale = 255 / iscale
            byte_im = (image) * scale
            byte_im = (byte_im.clip(0, 255) + 0.5).astype(np.uint8)
            image = byte_im
        # show image
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        im = ax.imshow(image)
        return im

    def show_mask(self, mask, ax=None, add_legend=True):
        """Show a a segmentation mask."""
        show_mask = np.empty((*mask.shape, 3))
        for c, color in self.CLASSES_COLORPALETTE.items():
            show_mask[mask == c, :] = color / 255.
        if ax is None:
            fig, ax = plt.subplots(figsizee=(10, 10))
        ax.axis("off")
        im = ax.imshow(show_mask)
        if add_legend:
            # show legend mapping pixel colors to class names
            import matplotlib.patches as mpatches
            handles = []
            for c in self.CLASSES_COLORPALETTE:
                handles.append(mpatches.Patch(color=self.CLASSES_COLORPALETTE[c] / 255., label=self.CLASSES_COLORPALETTE[c]))

            ax.legend(handles=handles)
        return im


    def compute_class_counts(self, set='train'):
        """Return the cumulated class counts for all masks in the training set."""
        assert set in ('train', 'test')
        if set == 'train':
            masks_paths = getattr(self, f'{set}_masks_paths')

        cumhist = np.zeros((self.N_CLASSES,), dtype=np.int64)
        for path in tqdm(masks_paths):
            with TiffFile(path) as tif:
                arr = tif.asarray()
                hist, _ = np.histogram(arr, bins=self.N_CLASSES, range=(0, 10)) # @todo: use bincount
                cumhist += hist
        return cumhist

    @dataclass
    class Item:
        """A class to manipulate data samples:
        path (Path): the abs. path to the file
        name (str): the name of the file (including .tif)
        id (int): an assigned ID, unique considering all train and test samples
        is_train (bool): sample is in train or not
        """
        path: str
        name: str
        id: int = None
        is_train: bool = True

    def get_train_test_tuples(self):
        """Get the train and test image items list of samples
        Args:
            The list of paths to files (can be input images or masks)
        Returns:
            (list[Item], list[Item]): list of image tuples for the train and test sets
        """
        # namedtuple containing image name (tiff file), id starting from 1, and a flag indicating if it is train or not
        train_images = [self.Item(p, p.name, is_train=True) for p in self.train_images_paths]
        test_images = [self.Item(p, p.name, is_train=False) for p in self.test_images_paths]
        images = sorted(train_images + test_images, key=lambda t: t.name)
        # set id by increasing order
        for id, item in enumerate(images, 1):
            item.id = id
        self.train_items = [t for t in images if t.is_train]
        self.test_items = [t for t in images if not t.is_train]
        return self.train_items, self.test_items


def numpy_parse_image(image_path):
    """Load an image and its segmentation mask as numpy arrays and returning a tuple
    Args:
        image_path (bytes): path to image
    Returns:
        (numpy.array[uint16], numpy.array[uint8]): the image and mask arrays
    """
    image_path = Path(bytes.decode(image_path))
    # get mask path from image path:
    # image should be in a imgs/IMAGE_ID.tif subfolder, while the mask is at masks/IMAGE_ID.tif
    mask_path = image_path.parent.parent/'masks'/image_path.name
    with TiffFile(image_path) as tifi, TiffFile(mask_path) as tifm:
        image = tifi.asarray()
        mask = tifm.asarray()
        # add channel dimension to mask: (256, 256, 1)
        mask = mask[..., None]
    return image, mask

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def parse_image(image_path):
    """Wraps the parse_image function as a TF function"""
    image, mask = tf.numpy_function(numpy_parse_image, (image_path,), (tf.uint16, tf.uint8))
    image.set_shape([LandCoverData.IMG_SIZE, LandCoverData.IMG_SIZE, LandCoverData.N_CHANNELS])
    mask.set_shape([LandCoverData.IMG_SIZE, LandCoverData.IMG_SIZE, 1])
    return image, mask


@tf.function
def normalize(input_image, input_mask):
    """Rescale the pixel values of the images between 0.0 and 1.0"""
    image = tf.cast(input_image, tf.float32) / LandCoverData.TRAIN_PIXELS_MAX
    return image, input_mask

@tf.function
def load_image_train(input_image, input_mask):
    """Apply optional augmentations and normalize a train image and its label mask."""

    image, mask = input_image, input_mask
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
        mask = tf.image.rot90(mask)
    elif tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=3)
        mask = tf.image.rot90(mask, k=3)

    image, mask = normalize(image, mask)
    return image, mask


@tf.function
def load_image_test(input_image, input_mask):
    """Normalize test image and its label mask."""
    image, mask = normalize(input_image, input_mask)
    return image, mask

if __name__ == '__main__':
    import multiprocessing
    N_CPUS = multiprocessing.cpu_count()

    DATA_FOLDER_STR = '~/challenge-ens/data'
    DATA_FOLDER = Path(DATA_FOLDER_STR).expanduser()
    dataset_folder = DATA_FOLDER/'v1_dataset'

    train_images_paths = sorted(list(dataset_folder.glob('train/imgs/*.tif')))
    test_images_paths = sorted(list(dataset_folder.glob('test/imgs/*.tif')))
    assert len(train_images_paths) == LandCoverData.TRAINSET_SIZE
    assert len(test_images_paths) == LandCoverData.TESTSET_SIZE

    train_dataset = tf.data.Dataset.list_files(str(dataset_folder/'train/imgs/*.tif'), seed=SEED)\
        .map(parse_image, num_parallel_calls=N_CPUS)
    test_dataset = tf.data.Dataset.list_files(str(dataset_folder/'test/imgs/*.tif'), seed=SEED)\
        .map(parse_image, num_parallel_calls=N_CPUS)
    # Test loading of data into tensors
    for (image, mask) in train_dataset.take(5):
        print({'image': image, 'mask': mask})

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    # datasets = {'train': train_dataset, 'test': test_dataset}

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
        .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SEED)\
        .repeat()\
        .batch(BATCH_SIZE)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.map(load_image_test)\
        .repeat()\
        .batch(BATCH_SIZE)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
