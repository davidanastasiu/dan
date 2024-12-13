import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true) + 1, np.array(pred) + 1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(model, pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape  # , mspe


def metric_g(name, pre, gt):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre) / 288)
    mae_all = []  # unused?
    mse_all = []  # unused?
    rmse_all = []
    mape_all = []
    l2 = []
    l3 = []
    lll = []
    for i in range(ll):
        mae, mse, rmse, mape = metric(
            name, pre[i * 288: (i + 1) * 288], gt[i * 288: (i + 1) * 288]
        )
        rmse_all.append(rmse)
        mape_all.append(mape)
    l2.append(np.around(np.mean(np.array(rmse_all)), 2))
    l3.append(np.around(np.mean(np.array(mape_all)), 3))
    lll.append(l2)
    lll.append(l3)
    return lll
