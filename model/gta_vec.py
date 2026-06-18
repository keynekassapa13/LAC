"""Vectorized soft-DTW for GTA (drop-in for gta.smoothDTW, dtw_prob path).

Sweeps anti-diagonals instead of the per-cell Python loop+clone, so it runs
on-GPU in float32 and avoids the /dev/shm blowup. Same output.
"""

import torch
import torch.nn.functional as F

def smoothDTW_vec(embs1, embs2, distance_type, softning, gamma_s, gamma_f):
    """Vectorized soft-DTW.  Returns (sdtw, dist) like the original."""
    if distance_type != "cosine":
        raise NotImplementedError(
            f"smoothDTW_vec only supports distance_type='cosine', got '{distance_type}'"
        )
    if softning != "dtw_prob":
        raise NotImplementedError(
            f"smoothDTW_vec only supports softning='dtw_prob', got '{softning}'"
        )

    # DP in float32 for AMP stability; downcast on return.
    orig_dtype = embs1.dtype
    embs1 = embs1.float()
    embs2 = embs2.float()
    gamma_s = float(gamma_s)
    gamma_f = float(gamma_f)

    # Cost matrix.
    dist = embs1 @ embs2.t()                          # (nrows, ncols)
    dist = -torch.log(F.softmax(dist / gamma_f, dim=0))

    nrows, ncols = dist.shape
    device = dist.device

    # Smooth infinity: softmax(-INF/gamma)->0 and 0*INF->0 in float32 (not NaN).
    INF = 1e9

    sdtw = torch.full((nrows + 1, ncols + 1), INF, device=device, dtype=torch.float32)
    # sdtw[0, 0] = 0 (functional update for autograd).
    zero_idx = torch.zeros(1, dtype=torch.long, device=device)
    sdtw = sdtw.index_put((zero_idx, zero_idx),
                          torch.zeros(1, device=device, dtype=torch.float32))

    # Anti-diagonal sweep: cells (i, j) with i+j=k are independent.
    for k in range(2, nrows + ncols + 1):
        i_lo = max(1, k - ncols)
        i_hi = min(nrows, k - 1)
        if i_hi < i_lo:
            continue

        i = torch.arange(i_lo, i_hi + 1, device=device)
        j = k - i  # (L,)

        # Predecessors on diagonals k-1 and k-2 (already computed).
        n_left = sdtw[i,     j - 1]
        n_diag = sdtw[i - 1, j - 1]
        n_up   = sdtw[i - 1, j]
        N_stack = torch.stack([n_left, n_diag, n_up], dim=0)         # (3, L)

        probs = F.softmax(-N_stack / gamma_s, dim=0)                  # (3, L)
        soft_min = (probs * N_stack).sum(dim=0)                       # (L,)
        new_val = dist[i - 1, j - 1] + soft_min                       # (L,)

        # Functional update (no in-place mutation, keeps the graph valid).
        sdtw = sdtw.index_put((i, j), new_val)

    if orig_dtype != torch.float32:
        sdtw = sdtw.to(orig_dtype)
        dist = dist.to(orig_dtype)

    return sdtw, dist
