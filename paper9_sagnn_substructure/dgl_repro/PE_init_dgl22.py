"""Positional encodings for SAGNN, original DGL version.

Only the DGL 2.x API differences are handled here: ``adjacency_matrix`` no
longer accepts ``scipy_fmt`` and returns a ``dgl.sparse.SparseMatrix``, and
``dgl.backend.asnumpy`` is gone.  The maths is unchanged.
"""

import dgl  # noqa: F401
import numpy as np
import torch
from scipy import sparse as sp


def _adjacency(g):
    mat = g.adjacency_matrix()
    if hasattr(mat, "to_dense"):
        return sp.csr_matrix(mat.to_dense().cpu().numpy())
    return sp.csr_matrix(mat)


def _degrees(g):
    deg = g.in_degrees()
    if isinstance(deg, torch.Tensor):
        return deg.cpu().numpy().astype(float)
    return np.asarray(deg, dtype=float)


def lap_positional_encoding(g, pos_enc_dim):
    """Graph positional encoding via Laplacian eigenvectors."""
    A = _adjacency(g).astype(float)
    N = sp.diags(np.clip(_degrees(g), 1, None) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N @ A @ N

    eig_val, eig_vec = np.linalg.eig(L.toarray())
    idx = eig_val.argsort()
    eig_vec = np.real(eig_vec[:, idx])

    pos_enc_emb = torch.from_numpy(eig_vec[:, 1:pos_enc_dim + 1]).float()
    if pos_enc_emb.shape[-1] < pos_enc_dim:
        offset = pos_enc_dim - pos_enc_emb.shape[-1]
        pos_enc_emb = torch.cat(
            (pos_enc_emb, torch.zeros(pos_enc_emb.shape[0], offset)), dim=-1
        )
    return pos_enc_emb


def init_positional_encoding(g, pos_enc_dim, type_init="rand_walk"):
    """Random-walk return probabilities."""
    n = g.number_of_nodes()
    if type_init != "rand_walk":
        raise ValueError(f"Unsupported positional encoding: {type_init}")

    A = _adjacency(g)
    Dinv = sp.diags(np.clip(_degrees(g), 1, None) ** -1.0, dtype=float)
    RW = A @ Dinv
    M = RW

    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(pos_enc_dim - 1):
        M_power = M_power @ M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    g.ndata["pos_enc"] = PE
    return PE
