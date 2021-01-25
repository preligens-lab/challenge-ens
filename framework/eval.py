"""
Evaluate the model prediction.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from framework.dataset import LandCoverData as LCD

def epsilon_kl_divergence(y_true, y_pred):
    """
        The metric of the challenge, the Kullback-Leibler divergence between two discrete distributions.
        Modification includes adding a small constant epsilon (1e-7) to y_true and y_pred
    Args:
            y_true (numpy.array[float]): ground-truth
            y_pred (numpy.arra[float]): prediction
    Returns:
        (float): the score (lower is better)
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)

    # Normalize to sum to 1 if it's not already
    y_true /= y_true.sum(1, keepdims=True)
    y_pred /= y_pred.sum(1, keepdims=True)
    # add a small constant for smoothness around 0
    y_true += 1e-7
    y_pred += 1e-7
    score = np.mean(np.sum(y_true * np.log(y_true / y_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score


def _parse_args():
    parser = argparse.ArgumentParser('Evaluation script')
    parser.add_argument('--gt-file', '-g', type=str, help="ground truth CSV file")
    parser.add_argument('--pred-file', '-p', type=str, required=True, help="prediction CSV file")
    parser.add_argument('--out-csv', '-o', type=str, default=None, help="output CSV file, optional")
    cli_args = parser.parse_args()
    cli_args.gt_file = Path(cli_args.gt_file).resolve()
    assert cli_args.gt_file.is_file()
    cli_args.pred_file = Path(cli_args.pred_file).resolve()
    assert cli_args.pred_file.is_file()
    return cli_args

if __name__ == '__main__':

    args = _parse_args()
    df_y_true = pd.read_csv(args.gt_file, index_col=0, sep=',')
    df_y_pred = pd.read_csv(args.pred_file, index_col=0, sep=',')

    if not len(df_y_true.index.intersection(df_y_pred.index)) == len(df_y_pred.index):
        raise ValueError("some samples IDs in y_pred are not present in y_true")

    # select subset of labels corresponding to predicted samples (train or val)
    df_y_true = df_y_true.loc[df_y_pred.index]

    # remove columns of the ignored classes if present
    ignored_classes = [LCD.CLASSES[i] for i in LCD.IGNORED_CLASSES_IDX]
    df_y_pred = df_y_pred.drop(ignored_classes, axis=1, errors='ignore')
    df_y_true = df_y_true.drop(ignored_classes, axis=1, errors='ignore')

    # use same order for columns
    df_y_pred = df_y_pred.loc[:, df_y_true.columns]

    if not df_y_pred.shape == df_y_true.shape:
        raise ValueError(f"shapes not equal between y_true and y_pred")

    if np.any(np.allclose(df_y_pred.sum(1), 0.)):
        # if any row predicted sums to 0, raise an error
        raise ValueError("some row vector(s) in y_pred sum to 0")

    score = epsilon_kl_divergence(df_y_true, df_y_pred)
    print(f"Score (mean Kullback-Leibler divergence) = \n{score}")
    if args.out_csv is not None:
        print(f"Saving evaluation CSV to file {args.out_csv}")
        df = pd.Series({
            'gt_file':args.gt_file,
            'pred_file':args.pred_file,
            'score': score
        }).to_frame().T
        df.to_csv(args.out_csv, index=False)
