"""
CAQTrainer: implements Algorithm 1 from Appendix B of the paper.

Algorithm 1 (exact reproduction):
    Require: M_PT, M_FT, calibration set D, top-k k, contrastive weight α, quantizer Q
    1:  Initialize transformation parameters θ
    2:  for each input x ∈ D do
    3:      Compute output distributions p_FT(y|x) and p_PT(y|x)
    4:      Apply transformation T_θ to M_FT to get p_Q(y|x)
    5:      Select vocabulary index sets:
    6:          S_top(x) ← top-k indices of p_FT(y|x)
    7:          S_diff(x) ← top-k indices of |p_FT(y|x) − p_PT(y|x)|
    8:      Renormalize p_FT^{S_top}, p_Q^{S_top}, p_PT^{S_diff}, p_Q^{S_diff} (Eq. 4)
    9:      Compute and accumulate loss:
    10:         L_CAL = KL(p_FT^{S_top} || p_Q^{S_top}) − α · KL(p_PT^{S_diff} || p_Q^{S_diff})
    11:     Update θ to minimize L_CAL
    12: end for
    13: Apply quantizer: M_Q ← Q(T_θ(M_FT))
    14: return M_Q
"""

import logging
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CAQConfig
from .loss import ContrastiveAlignmentLoss
from .models import ModelPair
from .transformation import SmoothScaleTransform
from .utils import clear_memory

logger = logging.getLogger(__name__)


class CAQTrainer:
    """
    Implements Algorithm 1: optimize transformation parameters θ using CAL.
    """

    def __init__(
        self,
        model_pair: ModelPair,
        transform: SmoothScaleTransform,
        loss_fn: ContrastiveAlignmentLoss,
        config: CAQConfig,
    ):
        self.model_pair = model_pair
        self.transform = transform
        self.loss_fn = loss_fn
        self.config = config

        # Adam optimizer on the scale parameters (paper: gradient-based optimization)
        self.optimizer = optim.Adam(
            transform.parameters_to_optimize(),
            lr=config.learning_rate,
        )

        # Move transform parameters to same device as M_FT
        ft_device = model_pair.get_ft_device()
        self.transform = self.transform.to(ft_device)

    def train(self, dataloader: DataLoader) -> dict:
        """
        Runs the CAL optimization loop over the calibration dataset.
        Implements lines 2-12 of Algorithm 1.

        Returns a dict with training statistics.
        """
        self.transform.train()

        total_loss = 0.0
        total_l_kl_top = 0.0
        total_l_cont_top = 0.0
        num_steps = 0

        pbar = tqdm(dataloader, desc="CAQ Training (CAL optimization)")

        for batch in pbar:
            input_ids = batch["input_ids"]  # (1, seq_len)

            # Lines 3: Compute p_FT(y|x) and p_PT(y|x) — no gradient
            # These are fixed reference distributions for this optimization step
            logits_ft = self.model_pair.get_logits_ft(input_ids)  # (seq_len, vocab)
            logits_pt = self.model_pair.get_logits_pt(input_ids)  # (seq_len, vocab)

            # Line 4: Apply T_θ to M_FT → p_Q(y|x) — with gradient
            # Hooks apply s^{-1} to activations, giving the transformed distribution
            logits_q = self.model_pair.get_logits_transformed(
                input_ids, self.transform
            )  # (seq_len, vocab), has grad

            # Lines 5-10: Compute L_CAL
            # (vocabulary subset selection and renormalization happen inside loss_fn)
            self.optimizer.zero_grad()
            loss, stats = self.loss_fn(logits_ft, logits_pt, logits_q)

            # Line 11: Update θ
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics
            total_loss += stats["l_cal"]
            total_l_kl_top += stats["l_kl_top"]
            total_l_cont_top += stats["l_cont_top"]
            num_steps += 1

            pbar.set_postfix({
                "loss": f"{stats['l_cal']:.4f}",
                "kl_top": f"{stats['l_kl_top']:.4f}",
                "cont": f"{stats['l_cont_top']:.4f}",
            })

            # Free intermediate tensors to keep GPU memory manageable
            del logits_ft, logits_pt, logits_q, loss
            clear_memory()

        avg_loss = total_loss / max(num_steps, 1)
        avg_kl_top = total_l_kl_top / max(num_steps, 1)
        avg_cont = total_l_cont_top / max(num_steps, 1)

        logger.info(
            f"CAL training complete. "
            f"Avg L_CAL={avg_loss:.4f}, "
            f"Avg L_KL-top={avg_kl_top:.4f}, "
            f"Avg L_cont-top={avg_cont:.4f}"
        )

        return {
            "avg_loss": avg_loss,
            "avg_l_kl_top": avg_kl_top,
            "avg_l_cont_top": avg_cont,
            "num_steps": num_steps,
        }

    def finalize(self):
        """
        Line 13 preparation: fuses θ into M_FT weights and releases M_PT.
        Returns the fused M_FT model, ready for quantization.
        """
        self.model_pair.fuse_and_release_pt(self.transform)
        return self.model_pair.get_fused_model()
