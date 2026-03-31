"""
Contrastive Alignment Loss (CAL) — paper Equations 5-7, 13-15.

Three distributions per input x at each token position:
    p_FT(y|x)  — safe fine-tuned model (pull target)
    p_PT(y|x)  — unsafe pre-trained model (push reference)
    p_Q(y|x)   — current quantized model under transformation θ (optimized)

Two vocabulary subsets (paper Section 3.2):
    S_top  = top-k indices of p_FT           → captures utility/coherence signal
    S_diff = top-k indices of |p_FT - p_PT|  → captures safety/alignment delta

Renormalized sub-distribution (Eq. 4):
    p^S(y) = p(y) / Σ_{y'∈S} p(y')

Loss (Eq. 15):
    L_CAL = L_KL-top - α * L_cont-top
where:
    L_KL-top   = KL(p_FT^{S_top}  || p_Q^{S_top})   → Eq. 13, pull toward safe
    L_cont-top = KL(p_PT^{S_diff} || p_Q^{S_diff})  → Eq. 14, push away from unsafe

KL direction: forward KL(reference || p_Q) ensures gradient flows through log(p_Q),
making the optimizer increase p_Q on tokens where reference has mass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    """
    Computes L_CAL = L_KL-top - alpha * L_cont-top averaged over token positions.

    Only logits_q should have requires_grad=True.
    logits_ft and logits_pt are treated as fixed reference distributions.
    """

    def __init__(self, top_k: int = 500, alpha: float = 0.75, eps: float = 1e-10):
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.eps = eps

    def _get_top_k_indices(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Returns indices of the top-k highest probabilities in p_FT.
        This is S_top: captures the model's intended utility and coherence.
        probs: (vocab_size,)  →  returns: (k,)
        """
        return torch.topk(probs, self.top_k, dim=-1).indices

    def _get_diff_indices(
        self, probs_ft: torch.Tensor, probs_pt: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns indices of top-k largest |p_FT - p_PT|.
        This is S_diff: isolates the 'behavioral delta' where safety alignment
        has most significantly shifted the model's prior distribution.
        probs_ft, probs_pt: (vocab_size,)  →  returns: (k,)
        """
        abs_diff = torch.abs(probs_ft - probs_pt)
        return torch.topk(abs_diff, self.top_k, dim=-1).indices

    def _renormalize(
        self, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Eq. 4: Extracts the subset at indices and renormalizes to sum=1.
        probs: (vocab_size,), indices: (k,)  →  returns: (k,)

        Note: index selection from probs is non-differentiable (straight-through
        w.r.t. the selected *values*, not the selection itself). Gradients flow
        through the values at the selected indices only.
        """
        subset = probs[indices]
        return subset / (subset.sum() + self.eps)

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Forward KL divergence: KL(p || q) = Σ p(y) * log(p(y) / q(y))
        p: reference distribution (detached), q: model distribution (has grad)
        Both are renormalized subsets of shape (k,).

        Uses clamping on q to avoid log(0). p is already a proper distribution
        (renormalized), so no clamping needed there.
        """
        q_clamped = torch.clamp(q, min=self.eps)
        p_clamped = torch.clamp(p, min=self.eps)
        return (p_clamped * (torch.log(p_clamped) - torch.log(q_clamped))).sum()

    def _compute_single_position(
        self,
        p_ft: torch.Tensor,  # (vocab_size,) — detached
        p_pt: torch.Tensor,  # (vocab_size,) — detached
        p_q: torch.Tensor,   # (vocab_size,) — has grad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Core per-token CAL computation. Returns (l_kl_top, l_cont_top).
        """
        # Select vocabulary subsets
        s_top = self._get_top_k_indices(p_ft)
        s_diff = self._get_diff_indices(p_ft, p_pt)

        # Pull component: KL(p_FT^{S_top} || p_Q^{S_top})  — Eq. 13
        pft_top = self._renormalize(p_ft, s_top)
        pq_top = self._renormalize(p_q, s_top)
        l_kl_top = self._kl_divergence(pft_top.detach(), pq_top)

        # Push component: KL(p_PT^{S_diff} || p_Q^{S_diff})  — Eq. 14
        ppt_diff = self._renormalize(p_pt, s_diff)
        pq_diff = self._renormalize(p_q, s_diff)
        l_cont_top = self._kl_divergence(ppt_diff.detach(), pq_diff)

        return l_kl_top, l_cont_top

    def forward(
        self,
        logits_ft: torch.Tensor,  # (seq_len, vocab_size) — no grad
        logits_pt: torch.Tensor,  # (seq_len, vocab_size) — no grad
        logits_q: torch.Tensor,   # (seq_len, vocab_size) — has grad
    ) -> tuple[torch.Tensor, dict]:
        """
        Computes L_CAL averaged over all token positions.

        Returns: (loss_scalar, {"l_kl_top": float, "l_cont_top": float})

        Processes positions in a loop to avoid (seq_len, vocab, vocab) intermediates.
        For seq_len=2048 and vocab=128K, batching over positions is memory-prohibitive.
        """
        seq_len = logits_ft.shape[0]

        # Convert to probabilities — detach references to avoid gradient leakage
        p_ft_all = F.softmax(logits_ft.detach().float(), dim=-1)  # (seq_len, vocab)
        p_pt_all = F.softmax(logits_pt.detach().float(), dim=-1)  # (seq_len, vocab)
        p_q_all = F.softmax(logits_q.float(), dim=-1)             # (seq_len, vocab), has grad

        total_l_kl_top = torch.tensor(0.0, device=logits_q.device)
        total_l_cont_top = torch.tensor(0.0, device=logits_q.device)

        for t in range(seq_len):
            l_kl_top, l_cont_top = self._compute_single_position(
                p_ft_all[t], p_pt_all[t], p_q_all[t]
            )
            total_l_kl_top = total_l_kl_top + l_kl_top
            total_l_cont_top = total_l_cont_top + l_cont_top

        # Average over token positions (Eq. 5-6 sum over D, we average per-sample)
        l_kl_top_mean = total_l_kl_top / seq_len
        l_cont_top_mean = total_l_cont_top / seq_len

        # Eq. 15: L_CAL = L_KL-top - alpha * L_cont-top
        loss = l_kl_top_mean - self.alpha * l_cont_top_mean

        return loss, {
            "l_kl_top": l_kl_top_mean.item(),
            "l_cont_top": l_cont_top_mean.item(),
            "l_cal": loss.item(),
        }
