"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_morris_gradient(model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Calculate gradients w.r.t static network inputs.

    TODO: Update Docstring with ref to paper
    
        Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.
    
    Returns
    -------
    torch.Tensor
        [description]
    """
    model.eval()
    grads = []
    for x_d, x_s, y in loader:
        model.zero_grad()
        x_s = torch.autograd.Variable(x_s, requires_grad=True)
        x_d, x_s = x_d.to(DEVICE), x_s.to(DEVICE)
        p = model(x_d, x_s[:, 0, :])[0]
        grad = torch.autograd.grad(p,
                                   x_s,
                                   grad_outputs=torch.ones_like(y).to(DEVICE),
                                   create_graph=False)
        grads.append(grad[0][:, 0, :].detach().cpu().numpy())
    return np.concatenate(grads, axis=0)
