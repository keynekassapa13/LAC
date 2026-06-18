"""Vectorized smooth Smith-Waterman with affine gaps.

Drop-in for lac._SoftSW. Sweeps anti-diagonals (O(N+M) vector ops on-device)
instead of the O(N*M) Python loop; same forward output, gradient-correct
backward, AMP/device clean.
"""

import torch


class _VectorizedSoftSW(torch.autograd.Function):
    """Vectorised forward + backward for the soft Smith-Waterman score."""

    @staticmethod
    def forward(ctx, S_xy, go, ge, temperature):
        N, M = S_xy.shape
        device = S_xy.device
        dtype = S_xy.dtype

        # Remember go/ge origin device/dtype (LAC keeps them on CPU);
        # gradients must be returned there.
        ctx.go_device = go.device
        ctx.ge_device = ge.device
        ctx.go_dtype  = go.dtype
        ctx.ge_dtype  = ge.dtype

        go = go.to(device=device, dtype=dtype)
        ge = ge.to(device=device, dtype=dtype)

        # Smooth -inf: exp(NEG/tau)->0 in fp32 without NaN.
        NEG = -1e9

        D  = torch.full((N + 1, M + 1), NEG, device=device, dtype=dtype)
        Ix = torch.full((N + 1, M + 1), NEG, device=device, dtype=dtype)
        Iy = torch.full((N + 1, M + 1), NEG, device=device, dtype=dtype)

        # Per-cell softmax probas saved for backward.
        D_p  = torch.zeros((N + 1, M + 1, 4), device=device, dtype=dtype)
        Ix_p = torch.zeros((N + 1, M + 1, 2), device=device, dtype=dtype)
        Iy_p = torch.zeros((N + 1, M + 1, 3), device=device, dtype=dtype)

        # In-place updates are safe here; backward is manual.
        with torch.no_grad():
            for k in range(2, N + M + 1):
                i_lo = max(1, k - M)
                i_hi = min(N, k - 1)
                if i_hi < i_lo:
                    continue

                i = torch.arange(i_lo, i_hi + 1, device=device)
                j = k - i  # j stays inside [1, M] by construction

                S_d  = S_xy[i - 1, j - 1]
                go_d = go[i - 1, j - 1]
                ge_d = ge[i - 1, j - 1]

                # --- D update (diagonal k-2) -----------------------------
                D_d  = D[i - 1, j - 1]
                Ix_d = Ix[i - 1, j - 1]
                Iy_d = Iy[i - 1, j - 1]
                zero = torch.zeros_like(D_d)
                D_stack = torch.stack([zero, D_d, Ix_d, Iy_d], dim=0)
                D_lse = temperature * torch.logsumexp(D_stack / temperature, dim=0)
                D_sm  = torch.softmax(D_stack / temperature, dim=0).t()  # (L, 4)

                D[i, j]    = S_d + D_lse
                D_p[i, j]  = D_sm

                # --- Ix update (diagonal k-1) ----------------------------
                D_l  = D[i, j - 1]
                Ix_l = Ix[i, j - 1]
                Ix_stack = torch.stack([D_l - go_d, Ix_l - ge_d], dim=0)
                Ix_lse = temperature * torch.logsumexp(Ix_stack / temperature, dim=0)
                Ix_sm  = torch.softmax(Ix_stack / temperature, dim=0).t()  # (L, 2)

                Ix[i, j]   = Ix_lse
                Ix_p[i, j] = Ix_sm

                # --- Iy update (diagonal k-1) ----------------------------
                D_u  = D[i - 1, j]
                Ix_u = Ix[i - 1, j]
                Iy_u = Iy[i - 1, j]
                Iy_stack = torch.stack([D_u - go_d, Ix_u - go_d, Iy_u - ge_d], dim=0)
                Iy_lse = temperature * torch.logsumexp(Iy_stack / temperature, dim=0)
                Iy_sm  = torch.softmax(Iy_stack / temperature, dim=0).t()  # (L, 3)

                Iy[i, j]   = Iy_lse
                Iy_p[i, j] = Iy_sm

        D_flat = D.flatten()
        value  = temperature * torch.logsumexp(D_flat / temperature, dim=0)
        probas = torch.softmax(D_flat / temperature, dim=0).reshape(N + 1, M + 1)

        ctx.save_for_backward(S_xy, D_p, Ix_p, Iy_p, probas)
        ctx.temperature = temperature
        # D[1:, 1:] = "logits" consumed by the cross-entropy term outside.
        return value, D[1:, 1:]

    @staticmethod
    def backward(ctx, grad_value, grad_D_out):
        S_xy, D_p, Ix_p, Iy_p, probas = ctx.saved_tensors
        temperature = ctx.temperature
        N, M = S_xy.shape
        device = S_xy.device
        dtype = S_xy.dtype

        # Pad one row/col so [i+1, j+1] indexing stays in bounds.
        grad_D  = torch.zeros((N + 2, M + 2), device=device, dtype=dtype)
        grad_Ix = torch.zeros((N + 2, M + 2), device=device, dtype=dtype)
        grad_Iy = torch.zeros((N + 2, M + 2), device=device, dtype=dtype)

        D_p_pad  = torch.zeros((N + 2, M + 2, 4), device=device, dtype=dtype)
        Ix_p_pad = torch.zeros((N + 2, M + 2, 2), device=device, dtype=dtype)
        Iy_p_pad = torch.zeros((N + 2, M + 2, 3), device=device, dtype=dtype)
        D_p_pad[: N + 1, : M + 1, :]  = D_p
        Ix_p_pad[: N + 1, : M + 1, :] = Ix_p
        Iy_p_pad[: N + 1, : M + 1, :] = Iy_p

        with torch.no_grad():
            # Reverse anti-diagonal sweep (reads diagonals k+1 and k+2).
            for k in range(N + M, 1, -1):
                i_lo = max(1, k - M)
                i_hi = min(N, k - 1)
                if i_hi < i_lo:
                    continue
                i = torch.arange(i_lo, i_hi + 1, device=device)
                j = k - i

                grad_Iy[i, j] = (
                    grad_D[i + 1, j + 1] * D_p_pad[i + 1, j + 1, 3]
                    + grad_Iy[i + 1, j] * Iy_p_pad[i + 1, j, 2]
                )

                grad_Ix[i, j] = (
                    grad_D[i + 1, j + 1] * D_p_pad[i + 1, j + 1, 2]
                    + grad_Ix[i, j + 1] * Ix_p_pad[i, j + 1, 1]
                    + grad_Iy[i + 1, j] * Iy_p_pad[i + 1, j, 1]
                )

                grad_D[i, j] = (
                    grad_D[i + 1, j + 1] * D_p_pad[i + 1, j + 1, 1]
                    + grad_Ix[i, j + 1] * Ix_p_pad[i, j + 1, 0]
                    + grad_Iy[i + 1, j] * Iy_p_pad[i + 1, j, 0]
                    + probas[i, j] * grad_value
                    + grad_D_out[i - 1, j - 1]
                )

        # dD[i,j]/dS[i-1,j-1] = 1, so grad_S = interior of grad_D.
        grad_S_xy = grad_D[1 : N + 1, 1 : M + 1].clone()

        # go: Ix slot 0, Iy slots 0,1 (all -). ge: Ix slot 1, Iy slot 2 (all -).
        grad_go = torch.zeros(N, M, device=device, dtype=dtype)
        grad_ge = torch.zeros(N, M, device=device, dtype=dtype)

        with torch.no_grad():
            for k in range(2, N + M + 1):
                i_lo = max(1, k - M)
                i_hi = min(N, k - 1)
                if i_hi < i_lo:
                    continue
                i = torch.arange(i_lo, i_hi + 1, device=device)
                j = k - i

                grad_go[i - 1, j - 1] = (
                    -grad_Ix[i, j] * Ix_p[i, j, 0]
                    - grad_Iy[i, j] * (Iy_p[i, j, 0] + Iy_p[i, j, 1])
                )
                grad_ge[i - 1, j - 1] = (
                    -grad_Ix[i, j] * Ix_p[i, j, 1]
                    - grad_Iy[i, j] * Iy_p[i, j, 2]
                )

        # Return go/ge grads on their original device/dtype.
        grad_go = grad_go.to(device=ctx.go_device, dtype=ctx.go_dtype)
        grad_ge = grad_ge.to(device=ctx.ge_device, dtype=ctx.ge_dtype)

        return grad_S_xy, grad_go, grad_ge, None


class VectorizedSoftSW(torch.nn.Module):
    """Vectorized SoftSW with the same interface as ``lac.SoftSW``."""

    def __init__(self, go, ge, temperature=1.0):
        super().__init__()
        self.go = go
        self.ge = ge
        self.temperature = temperature
        self.func_dtw = _VectorizedSoftSW.apply

    def calc_distance_matrix(self, x, y):
        # Squared Euclidean cost. x: (B, N, D), y: (B, M, D)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return (x - y).pow(2).sum(3)

    def forward(self, x, y):
        S_xy = self.calc_distance_matrix(x, y)            # (B, N, M)
        B, N, M = S_xy.shape

        loss = 0
        probas = []
        eps = 1e-8
        for b in range(B):
            # SW similarity = 1 / distance (as in the original).
            S = 1.0 / (S_xy[b] + eps)
            value, D_out = self.func_dtw(S, self.go, self.ge, self.temperature)
            loss = loss + value
            probas.append(D_out)
        return loss / B, torch.stack(probas, 0)
