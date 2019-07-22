"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from typing import Dict, Tuple

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function
    
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_shape_collections(data: Dict):
    shapes = []

    for points in data.values():
        shapes.append(Polygon(np.array([points['lons'], points['lats']]).T, closed=True))

    collection = PatchCollection(shapes)
    collection.set_facecolor('#eeeeee')
    collection.set_edgecolor('black')
    collection.set_linewidth(0.2)

    return collection


model_draw_style = {
    'ealstm_NSE': {
        'ensemble_color': '#1b9e77',
        'single_color': '#b3e2cd',
        'linestyle': '-',
        'marker': 's',
        'label': 'EA-LSTM NSE'
    },
    'ealstm_MSE': {
        'ensemble_color': '#1b9e77',
        'single_color': '#b3e2cd',
        'linestyle': '--',
        'marker': 's',
        'label': 'EA-LSTM MSE'
    },
    'lstm_NSE': {
        'ensemble_color': '#d95f02',
        'single_color': '#fdcdac',
        'linestyle': '-',
        'marker': 's',
        'label': 'LSTM NSE'
    },
    'lstm_MSE': {
        'ensemble_color': '#d95f02',
        'single_color': '#fdcdac',
        'linestyle': '--',
        'marker': 's',
        'label': 'LSTM MSE'
    },
    'lstm_no_static_MSE': {
        'ensemble_color': '#7570b3',
        'single_color': '#cbd5e8',
        'linestyle': '--',
        'marker': '^',
        'label': 'LSTM (no static inputs) MSE'
    },
    'lstm_no_static_NSE': {
        'ensemble_color': '#7570b3',
        'single_color': '#cbd5e8',
        'linestyle': '-',
        'marker': '^',
        'label': 'LSTM (no static inputs) NSE'
    },
    'SAC_SMA': {
        'color': '#e7298a',
        'linestyle': '-.',
        'marker': None,
        'label': 'SAC-SMA'
    },
    'VIC_basin': {
        'color': '#66a61e',
        'linestyle': '-.',
        'marker': None,
        'label': 'VIC (basin-wise calibrated)'
    },
    'VIC_conus': {
        'color': '#66a61e',
        'linestyle': '-.',
        'marker': None,
        'label': 'VIC (CONUS-wide calibrated)'
    },
    'mHm_basin': {
        'color': '#e6ab02',
        'linestyle': '-.',
        'marker': None,
        'label': 'mHm (basin-wise calibrated)'
    },
    'mHm_conus': {
        'color': '#e6ab02',
        'linestyle': '-.',
        'marker': None,
        'label': 'mHm (CONUS-wide calibrated)'
    },
    'HBV_lb': {
        'color': '#a6761d',
        'linestyle': '-.',
        'marker': 'x',
        'label': 'HBV lower bound (n=1000 uncalibrated)'
    },
    'HBV_ub': {
        'color': '#a6761d',
        'linestyle': '-.',
        'marker': None,
        'label': 'HBV upper bound (n=100 calibrated)'
    },
    'q_sim_fuse_900': {
        'color': '#666666',
        'linestyle': '-.',
        'marker': None,
        'label': 'FUSE (900)'
    },
    'q_sim_fuse_902': {
        'color': '#666666',
        'linestyle': '-.',
        'marker': '.',
        'label': 'FUSE (902)'
    },
    'q_sim_fuse_904': {
        'color': '#666666',
        'linestyle': '-.',
        'marker': 'd',
        'label': 'FUSE (904)'
    }
}

model_specs = {
    'ealstm_MSE': {
        'model': 'ealstm',
        'loss': 'MSELoss'
    },
    'ealstm_NSE': {
        'model': 'ealstm',
        'loss': 'NSELoss'
    },
    'lstm_MSE': {
        'model': 'lstm',
        'loss': 'MSELoss'
    },
    'lstm_NSE': {
        'model': 'lstm',
        'loss': 'NSELoss'
    },
    'lstm_no_static_MSE': {
        'model': 'lstm_no_static',
        'loss': 'MSELoss'
    },
    'lstm_no_static_NSE': {
        'model': 'lstm_no_static',
        'loss': 'NSELoss'
    }
}

attribute_draw_style = {
    # soil features
    'silt_frac': {
        'color': '#ffffe5',
        'marker': 'v'
    },
    'soil_depth_pelletier': {
        'color': '#fff7bc',
        'marker': 'v'
    },
    'clay_frac': {
        'color': '#fee391',
        'marker': 'v'
    },
    'soil_conductivity': {
        'color': '#fec44f',
        'marker': 'v'
    },
    'max_water_content': {
        'color': '#fe9929',
        'marker': 'v'
    },
    'geol_permeability': {
        'color': '#ec7014',
        'marker': 'v'
    },
    'soil_porosity': {
        'color': '#cc4c02',
        'marker': 'v'
    },
    'sand_frac': {
        'color': '#993404',
        'marker': 'v'
    },
    'soil_depth_statsgo': {
        'color': '#662506',
        'marker': 'v'
    },
    'carbonate_rocks_frac': {
        'color': '#000000',
        'marker': 'v'
    },
    # climate indices
    'p_seasonality': {
        'color': '#fff7fb',
        'marker': 'o'
    },
    'low_prec_dur': {
        'color': '#ece7f2',
        'marker': 'o'
    },
    'aridity': {
        'color': '#d0d1e6',
        'marker': 'o'
    },
    'pet_mean': {
        'color': '#a6bddb',
        'marker': 'o'
    },
    'frac_snow': {
        'color': '#74a9cf',
        'marker': 'o'
    },
    'low_prec_freq': {
        'color': '#3690c0',
        'marker': 'o'
    },
    'p_mean': {
        'color': '#0570b0',
        'marker': 'o'
    },
    'high_prec_dur': {
        'color': '#045a8d',
        'marker': 'o'
    },
    'high_prec_freq': {
        'color': '#023858',
        'marker': 'o'
    },
    # vegetation properties
    'gvf_max': {
        'color': '#d9f0a3',
        'marker': '*'
    },
    'frac_forest': {
        'color': '#addd8e',
        'marker': '*'
    },
    'lai_max': {
        'color': '#78c679',
        'marker': '*'
    },
    'lai_diff': {
        'color': '#41ab5d',
        'marker': '*'
    },
    'gvf_diff': {
        'color': '#238443',
        'marker': '*'
    },
    # general
    'slope_mean': {
        'color': '#fcc5c0',
        'marker': 's'
    },
    'elev_mean': {
        'color': '#f768a1',
        'marker': 's'
    },
    'area_gages2': {
        'color': '#7a0177',
        'marker': 's'
    },
}
