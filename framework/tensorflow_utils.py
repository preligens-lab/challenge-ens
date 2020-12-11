from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
import tensorflow as tf

def display_sample(display_list):
    """Visualize side-by-side an input image, the ground-truth mask and the prediction mask
    Args:
        display_list (list[tf.Tensor or numpy.array]): of length 1, 2 or 3, containing the input
        image, the ground-truth mask and the prediction mask
    """
    fig, axs = plt.subplots(1, len(display_list), figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        axs[i].set_title(title[i])
        axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        axs[i].axis('off')


def create_mask(pred_raster: tf.Tensor) -> tf.Tensor:
    """Return a predicted mask with the top 1 predictions only.
    Args:
        pred_raster (tf.Tensor): a [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
            N_CLASS values (vector) which represents the predicted probability of the pixel
            belonging to these classes.
    Returns:
        (tf.Tensor): a [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions for each pixels.
    """
    pred_mask = tf.argmax(pred_raster, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def plot_predictions(model, dataset=None, sample_batch=None, num=1, save_filepaths=None):
    """Show a sample prediction.
    @ TOTEST
    @ TODO: docstring
    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if save_filepaths is not None:
        assert isinstance(save_filepaths, list) and len(save_filepaths) == num
    if dataset is not None:
        for i, (image, mask) in enumerate(dataset.take(num)):
            pred_raster = model.predict(image)
            display_sample([image[0], mask, create_mask(pred_raster)])
            if save_filepaths is not None:
                fig = plt.gcf()
                fig.savefig(save_filepaths[i], bbox_inches='tight', dpi=300)
    else:
        image, mask = sample_batch
        pred_raster = model.predict(image)
        pred_mask = create_mask(pred_raster)
        for i in range(num):
            display_sample([image[i], mask[i], pred_mask[i]])
            if save_filepaths is not None:
                fig = plt.gcf()
                fig.savefig(save_filepaths[i], bbox_inches='tight', dpi=300)
