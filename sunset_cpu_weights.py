"""CPU-only exact zero-skipping adapter for TorchModel's recurrent matmul.

TorchModel is imported unchanged. Its binary spike matrix is exceptionally sparse.
Selecting the columns whose presynaptic neurons fired avoids scanning 15M edges
on every CPU timestep. No connections, neurons, or nonzero contributions are
removed. All membrane, synaptic, delay, refractory and Poisson computations remain
in the original PyTorch model. The adapter does not support autograd or CUDA.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix

class CPUEventWeights:
    def __init__(self, weights):
        if weights.device.type != 'cpu':
            raise ValueError('CPU weights required')
        weights = weights.to_sparse_csr()
        self.matrix = csr_matrix((weights.values().numpy(),
                                  weights.col_indices().numpy(),
                                  weights.crow_indices().numpy()), shape=weights.shape).tocsc()
        self.shape = weights.shape

    def transpose(self, dim0, dim1):
        if (dim0, dim1) != (0, 1):
            raise ValueError('Only the TorchModel transpose is supported')
        return _TransposedWeights(self)

class _TransposedWeights:
    def __init__(self, owner):
        self.owner = owner

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is not torch.matmul:
            return NotImplemented
        spikes, transposed = args
        if spikes.device.type != 'cpu' or spikes.requires_grad:
            raise ValueError('CPU inference without gradients is required')
        array = spikes.detach().numpy()
        out = np.zeros((array.shape[0], transposed.owner.shape[0]), dtype=np.float32)
        for trial, row in enumerate(array):
            active = np.flatnonzero(row)
            if len(active):
                out[trial] = transposed.owner.matrix[:, active] @ row[active]
        return torch.from_numpy(out)
