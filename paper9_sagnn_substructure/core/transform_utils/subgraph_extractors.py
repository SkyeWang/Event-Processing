"""Subgraph extractors used by SAGNN.

The original file imported ``torch_sparse``/``torch_cluster`` for two unused
helpers.  Only the k-hop extractor is needed by the pipeline, so it is
re-implemented with SciPy and the compiled extensions are no longer required.
"""

import numpy as np
import torch
from scipy import sparse as sp


def k_hop_subgraph(edge_index, num_nodes, num_hops):
    """Return a dense node mask and hop indicator for every node's k-hop ego graph.

    Args:
        edge_index: ``[2, E]`` long tensor.
        num_nodes: number of nodes in the graph.
        num_hops: maximum hop distance.

    Returns:
        node_mask: ``[N, N]`` bool tensor, ``node_mask[v, u]`` is True when ``u``
            is within ``num_hops`` of ``v``.
        hop_indicator: ``[N, N]`` long tensor with the hop distance, ``-1`` when
            unreachable.
    """
    n = int(num_nodes)
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    adj = sp.coo_matrix(
        (np.ones(row.shape[0], dtype=np.float64), (row, col)), shape=(n, n)
    ).tocsr()

    hop_masks = [torch.eye(n, dtype=torch.bool)]
    hop_indicator = torch.full((n, n), -1, dtype=torch.long)
    hop_indicator[hop_masks[0]] = 0

    frontier = hop_masks[0].numpy().astype(np.float64)
    for i in range(num_hops):
        nxt = np.asarray((adj @ frontier) > 0)
        nxt_t = torch.from_numpy(nxt).bool()
        hop_masks.append(nxt_t)
        hop_indicator[(hop_indicator == -1) & nxt_t] = i + 1
        frontier = nxt.astype(np.float64)

    hop_indicator = hop_indicator.T
    node_mask = hop_indicator >= 0
    return node_mask, hop_indicator
