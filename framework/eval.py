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
    parser.add_argument('--gt-file', '-g', type=str, help="Ground truth CSV file")
    parser.add_argument('--pred-file', '-p', type=str, required=True, help="Prediction CSV file")

    cli_args = parser.parse_args()
    cli_args.gt_file = Path(cli_args.gt_file)
    assert cli_args.gt_file.is_file()
    cli_args.pred_file = Path(cli_args.pred_file)
    assert cli_args.pred_file.is_file()
    return cli_args

if __name__ == '__main__':

    args = _parse_args()
    df_y_true = pd.read_csv(args.gt_file, index_col=0, sep=',')
    df_y_pred = pd.read_csv(args.pred_file, index_col=0, sep=',')

    if not len(df_y_true.index.intersection(df_y_pred.index)) == len(df_y_true.index):
        raise ValueError("indexes are not equal")

    # use same order for rows
    df_y_pred = df_y_pred.loc[df_y_true.index]
    # remove ignored class columns "no_data" and "clouds" if present
    df_y_pred = df_y_pred.drop(['no_data', 'clouds'], axis=1, errors='ignore')
    df_y_true = df_y_true.drop(['no_data', 'clouds'], axis=1, errors='ignore')

    # use same order for columns
    df_y_pred = df_y_pred.loc[:, df_y_true.columns]

    if not df_y_pred.shape == df_y_true.shape == (LCD.TESTSET_SIZE, LCD.N_CLASSES-2):
        # @ TODO
        raise ValueError(f"shapes not equal to : {LCD.TESTSET_SIZE, LCD.N_CLASSES-2}")

    if np.any(np.allclose(df_y_pred.sum(1), 0.)):
        # if any row predicted sums to 0, raise an error
        raise ValueError("some row vector(s) in y_pred sum to 0")

    score = epsilon_kl_divergence(df_y_true, df_y_pred)
    print(f"(ε)KL divergence =  {score}")
