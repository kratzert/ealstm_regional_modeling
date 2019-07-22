"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import json
import pickle
from collections import defaultdict
from pathlib import PosixPath
from typing import Callable, Dict, List, Tuple

import numpy as np
import tqdm
import xarray
from scipy.stats import wilcoxon


def get_run_dirs(root_dir: PosixPath, model: str, loss: str) -> List:
    """Get all folders that are trained for a specific model configuration
    
    Parameters
    ----------
    root_dir : PosixPath
        Path to the folder containing all model runs.
    model : str
        One of ['ealstm', 'lstm', 'lstm_no_static'], defining the model type to find.
    loss : str
        One of ['NSELoss', 'MSELoss'], defining the loss function that the model was trained for.
    
    Returns
    -------
    List
        List of PosixPaths, where each path points to the folder of one model run.
    
    Raises
    ------
    ValueError
        If an invalid model type was passed.
    ValueError
        If an invalid loss type was passed.
    RuntimeError
        If root directory contains no subfolder.
    """
    valid_models = ["ealstm", "lstm", "lstm_no_static"]
    if not model in valid_models:
        raise ValueError(f"`model` must be one of {valid_models}")

    valid_loss = ['MSELoss', 'NSELoss']
    if not loss in valid_loss:
        raise ValueError(f"`loss` must be one of {valid_loss}")

    folders = list(root_dir.glob('*/'))

    if len(folders) == 0:
        raise RuntimeError(f"No subfolders found in {root_dir}")

    run_dirs = []
    for folder in folders:
        if folder.is_dir():
            with open(folder / "cfg.json", "r") as fp:
                cfg = json.load(fp)

            if (model == "ealstm") and (not cfg["concat_static"]) and (not cfg["no_static"]):
                if (loss == "NSELoss") and (not cfg["use_mse"]):
                    run_dirs.append(folder)
                elif (loss == "MSELoss") and (cfg["use_mse"]):
                    run_dirs.append(folder)
                else:
                    pass

            if (model == "lstm") and (cfg["concat_static"]) and (not cfg["no_static"]):
                if (loss == "NSELoss") and (not cfg["use_mse"]):
                    run_dirs.append(folder)
                elif (loss == "MSELoss") and (cfg["use_mse"]):
                    run_dirs.append(folder)
                else:
                    pass

            if (model == "lstm_no_static") and (cfg["no_static"]):
                if (loss == "NSELoss") and (not cfg["use_mse"]):
                    run_dirs.append(folder)
                elif (loss == "MSELoss") and (cfg["use_mse"]):
                    run_dirs.append(folder)
                else:
                    pass

    return run_dirs


def eval_benchmark_models(netcdf_folder: PosixPath, func: Callable) -> dict:
    """Evaluate benchmark models on specific metric function.
    
    Parameters
    ----------
    netcdf_folder : PosixPath
        Directory, containing basin-wise netcdf files, which contain the benchmark model simulations
    func : Callable
        The metric function to evaluate. Must satisfy the func(obs, sim) convention.
    
    Returns
    -------
    dict
        Dictionary, containing the metric values of each basin and each benchmark model.
    """
    nc_files = list(netcdf_folder.glob('*.nc'))
    benchmark_models = defaultdict(dict)
    for nc_file in tqdm.tqdm(nc_files):
        basin = nc_file.name[:8]
        xr = xarray.open_dataset(nc_file)
        for key in xr.keys():
            if key != 'QObs':
                obs = xr['QObs'].values
                sim = xr[key].values
                sim = sim[obs >= 0]
                obs = obs[obs >= 0]
                value = func(obs, sim)
                if np.isnan(value):
                    print(f"{key}: {nc_file}")
                else:
                    benchmark_models[key][basin] = value
    return benchmark_models


def eval_lstm_models(run_dirs: List, func: Callable) -> dict:
    """Evaluate LSTM outputs on specific metric function.

    Returns the metric for each basin in each seed, as well as the results of the ensemble mean.
    
    Parameters
    ----------
    run_dirs : List
        List of PosixPaths pointing to the different model directories.
    func : Callable
        The metric function to evaluate. Must satisfy the func(obs, sim) convention.
    
    Returns
    -------
    dict
        Dictionary, containing the metric value for each basin of each random seed, as well as the 
        ensemble mean.
    """
    single_models = {}
    model_ensemble = defaultdict(dict)
    for run_dir in tqdm.tqdm(run_dirs):
        eval_file = list(run_dir.glob("*.p"))[0]
        parts = eval_file.name.split('_')
        seed = parts[-1][:-2]
        single_models[seed] = {}
        with eval_file.open("rb") as fp:
            data = pickle.load(fp)
        for basin, df in data.items():
            obs = df["qobs"].values
            sim = df["qsim"].values
            sim = sim[obs >= 0]
            obs = obs[obs >= 0]
            single_models[seed][basin] = func(obs, sim)
            if basin not in model_ensemble.keys():
                model_ensemble[basin]["df"] = df
            else:
                model_ensemble[basin]["df"]["qsim"] += df["qsim"]

    ensemble_nse = {}
    for basin, data in model_ensemble.items():
        obs = data["df"]["qobs"].values
        sim = data["df"]["qsim"].values / len(single_models.keys())
        sim = sim[obs >= 0]
        obs = obs[obs >= 0]
        ensemble_nse[basin] = func(obs, sim)

    single_models["ensemble"] = ensemble_nse

    return single_models


def get_pvals(metrics: dict, model1: str, model2: str) -> Tuple[List, float]:
    """[summary]
    
    Parameters
    ----------
    metrics : dict
        Dictionary, containing the metric values of both models for all basins.
    model1 : str
        String, defining the first model to take. Must be a key in `metrics`
    model2 : str
        String, defining the second model to take. Must be a key in `metrics`
    
    Returns
    -------
    p_vals : List
        List, containing the p-values of all possible seed combinations.
    p_val : float
        P-value between the ensemble means.
    """

    # p-values between mean performance per basin of both models
    metric_model1 = get_mean_basin_performance(metrics, model1)
    metric_model2 = get_mean_basin_performance(metrics, model2)
    _, p_val_single = wilcoxon(list(metric_model1.values()), list(metric_model2.values()))

    # p-value between ensemble means
    _, p_val_ensemble = wilcoxon(list(metrics[model1]["ensemble"].values()),
                                 list(metrics[model2]["ensemble"].values()))
    return p_val_single, p_val_ensemble


def get_mean_basin_performance(metrics: dict, model: str) -> Dict:
    """Get the mean performance per basin for a given model

    Parameters
    ----------
    metrics : dict
        Dictionary containing all evaluation metrics
    model : str
        Model identifier string

    Returns
    -------
    Dict
        Dictionary containing for each basin a key and the value is the mean performance.
    """
    seeds = [k for k in metrics[model].keys() if k != "ensemble"]
    metric = defaultdict(list)
    for seed in seeds:
        for basin, nse in metrics[model][seed].items():
            metric[basin].append(nse)
    return {basin: np.mean(values) for basin, values in metric.items()}


def get_cohens_d(values1: List, values2: List) -> float:
    """Calculate Cohen's Effect size
    
    Parameters
    ----------
    values1 : List
        List of model performances of model 1
    values2 : List
        List of model performances of model 2
    
    Returns
    -------
    float
        Cohen's d
    """
    s = np.sqrt(((len(values1) - 1) * np.var(values1) + (len(values2) - 1) * np.var(values2)) /
                (len(values1) + len(values2) - 2))
    d = (np.abs(np.mean(values1) - np.mean(values2))) / s
    return d
