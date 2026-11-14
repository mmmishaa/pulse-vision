import numpy as np

def mae(pred_hr: np.ndarray, target_hr: np.ndarray):
    if pred_hr.shape != target_hr.shape:
        raise ValueError("mismatch in the number of elements in the arrays")

    metric = np.mean(np.abs(pred_hr - target_hr))
    return metric
    

def rmse(pred_hr: np.ndarray, target_hr: np.ndarray):
    if pred_hr.shape != target_hr.shape:
        raise ValueError("mismatch in the number of elements in the arrays")

    metric = np.sqrt(np.mean((pred_hr - target_hr) ** 2))
    return metric

def bpm_pearson_corr(pred_hr: np.ndarray, target_hr: np.ndarray):
    if pred_hr.shape != target_hr.shape:
        raise ValueError("mismatch in the number of elements in the arrays")

    eps = 1e-8
    num = np.sum((pred_hr - pred_hr.mean()) * (target_hr - target_hr.mean()))
    den = np.sqrt(np.sum((pred_hr - pred_hr.mean()) ** 2) * np.sum((target_hr - target_hr.mean()) ** 2))
    metric = num / (den + eps)

    return metric