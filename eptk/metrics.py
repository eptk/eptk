# -*- coding: utf-8 -*-
from sklearn import metrics
import numpy as np
from sklearn.utils.validation import check_consistent_length, check_array

"""Metrics to assess performance on energy prediction task."""


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task.
    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().
    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.
    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.
    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.
    """
    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output "
            "({0}!={1})".format(y_true.shape[1], y_pred.shape[1])
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                "Allowed 'multioutput' string values are {}. "
                "You provided multioutput={!r}".format(
                    allowed_multioutput_str, multioutput
                )
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in " "multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(
                ("There must be equally many custom weights " "(%d) as outputs (%d).")
                #                 % (len(multioutput), n_outputs)
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean absolute percentage error regression loss.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)
    """
    if len(y_pred[y_pred < 0]):
        print("MAPE : Negative Predictions are adjusted to zero.")
        y_pred[y_pred < 0] = 0
    print("MAPE : Zero true values will be ignored")
    y = y_true[:][y_true[:] != 0]
    pred = y_pred[:][y_true[:] != 0]
    summ = np.sum(abs((y[:] - pred[:]) / y[:]))
    return summ / len(y)



def mean_absolute_error(y_true, y_predict):
    """mean absolute error loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)

        """
    if len(y_predict[y_predict < 0]):
        print("MAE : Negative Predictions are adjusted to zero.")
        y_predict[y_predict < 0] = 0

    mae = metrics.mean_absolute_error
    return mae(y_true, y_predict)


def mean_squared_error(y_true, y_predict):
    """ mean squared error loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)

        """

    if len(y_predict[y_predict < 0]):
        print("MSE : Negative Predictions are adjusted to zero.")
        y_predict[y_predict < 0] = 0

    mse = metrics.mean_squared_error
    return mse(y_true, y_predict)


def root_mean_squared_error(y_true, y_predict):
    """ root mean squared error loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)

        """

    if len(y_predict[y_predict < 0]):
        print("RMSE : Negative Predictions are adjusted to zero.")
        y_predict[y_predict < 0] = 0

    mse = metrics.mean_squared_error
    return mse(y_true, y_predict, squared=False)


def mean_squared_log_error(y_true, y_predict):
    """mean squared logarithmic error loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)

        """
    if len(y_predict[y_predict < 0]):
        print("MSLE : Negative Predictions are adjusted to zero.")
        y_predict[y_predict < 0] = 0
    msle = metrics.mean_squared_log_error
    return msle(y_true, y_predict)


def root_mean_squared_log_error(y_true, y_predict):
    """root mean squared logarithmic error loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
        A non-negative floating point value (the best value is 0.0)

        """

    if len(y_predict[y_predict < 0]):
        print("RMSLE : Negative Predictions are adjusted to zero.")
        y_predict[y_predict < 0] = 0
    msle = metrics.mean_squared_log_error
    return np.sqrt(msle(y_true, y_predict))
