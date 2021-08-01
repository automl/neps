import numpy as np


def _check_data(pred, target):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred).reshape(-1)
    if not isinstance(target, np.ndarray):
        target = np.array(target).reshape(-1)
    return pred, target


def rmse(pred, target) -> float:
    """Compute the root mean squared error"""
    pred, target = _check_data(pred, target)
    assert pred.shape[0] == target.shape[0], (
        "predictant shape "
        + str(pred.shape[0])
        + " but target shape"
        + str(target.shape[0])
    )
    n = pred.shape[0]
    return np.sqrt(np.sum((pred - target) ** 2) / n)


def nll(pred, pred_std, target) -> float:
    """Compute the negative log-likelihood (over the validation dataset)"""
    from scipy.stats import norm

    pred, target = _check_data(pred, target)
    total_nll_origin = -np.mean(norm.logpdf(target, loc=pred, scale=pred_std))
    return total_nll_origin


def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats

    pred, target = _check_data(pred, target)
    coef_val, _ = stats.spearmanr(pred, target)
    return coef_val


def average_error(pred, target, log_val=True) -> float:
    pred, target = _check_data(pred, target)
    if log_val:
        pred = np.exp(pred)
        target = np.exp(target)
    return np.mean(np.abs((pred - target))).item()
