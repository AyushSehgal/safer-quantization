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
        loss_fn: ContrastiveAlignmentLoss,
        config: CAQConfig,
    ):
        self.model_pair = model_pair
        self.loss_fn = loss_fn
        self.config = config
        # transform and optimizer are created in _precompute_pt_logits after M_FT loads
        self.transform: Optional[SmoothScaleTransform] = None
        self.optimizer: Optional[optim.Adam] = None

    def _precompute_pt_logits(self, dataloader: DataLoader) -> list:
        """
        Pre-computes M_PT logits for all calibration samples on GPU, then deletes
        M_PT and loads M_FT. This ensures the two 7B models never occupy GPU memory
        simultaneously, avoiding OOM on 32 GB GPUs at seq_len=2048.

        Sequence:
          1. M_PT is already on GPU (loaded first in ModelPair.load()).
          2. Run all calibration samples through M_PT; cache logits to CPU.
          3. Delete M_PT, freeing ~14 GB of GPU VRAM.
          4. Load M_FT onto the now-empty GPU.

        Logits are stored on CPU and moved to GPU one batch at a time during training.
        Cache is saved to disk so a restart after a crash skips steps 2-3.
        """
        import os
        cache_path = os.path.join(self.config.output_dir, "pt_logits_cache.pt")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached M_PT logits from {cache_path}")
            pt_logits_cache = torch.load(cache_path, map_location="cpu")
            logger.info("Loaded M_PT logits from cache. Skipping pre-computation.")
            # M_PT is no longer needed — release it and load M_FT
            del self.model_pair.model_pt
            self.model_pair.model_pt = None
            clear_memory()
            self.model_pair.load_ft()
        else:
            # M_PT is already on GPU — run forward passes and cache logits
            pt_logits_cache = []
            for batch in tqdm(dataloader, desc="Pre-computing M_PT logits (GPU)"):
                logits_pt = self.model_pair.get_logits_pt(batch["input_ids"])
                pt_logits_cache.append(logits_pt.cpu())

            logger.info(f"Saving M_PT logits cache to {cache_path}")
            torch.save(pt_logits_cache, cache_path)

            logger.info("Releasing M_PT — freeing GPU VRAM before loading M_FT...")
            del self.model_pair.model_pt
            self.model_pair.model_pt = None
            clear_memory()
            logger.info("M_PT released.")

            # Now GPU is free — load M_FT
            self.model_pair.load_ft()

        # M_FT is now on GPU — initialize transform and optimizer
        ft_device = self.model_pair.get_ft_device()
        self.transform = SmoothScaleTransform(self.model_pair.model_ft).to(ft_device)
        self.optimizer = optim.Adam(
            self.transform.parameters_to_optimize(),
            lr=self.config.learning_rate,
        )
        logger.info(
            f"M_FT loaded on {ft_device}. "
            f"Initialized {self.transform.num_parameters():,} transform parameters. "
            f"Pre-computation complete."
        )
        return pt_logits_cache

    def train(self, dataloader: DataLoader) -> dict:
        """
        Runs the CAL optimization loop over the calibration dataset.
        Implements lines 2-12 of Algorithm 1.

        Returns a dict with training statistics.
        """
        # Pre-compute all M_PT logits upfront so the training loop runs purely on GPU
        pt_logits_cache = self._precompute_pt_logits(dataloader)

        self.transform.train()

        total_loss = 0.0
        total_l_kl_top = 0.0
        total_l_cont_top = 0.0
        num_steps = 0

        ft_device = self.model_pair.get_ft_device()
        pbar = tqdm(dataloader, desc="CAQ Training (CAL optimization)")

        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"]  # (1, seq_len)

            # Lines 3: Compute p_FT(y|x) and p_PT(y|x) — no gradient
            # These are fixed reference distributions for this optimization step
            logits_ft = self.model_pair.get_logits_ft(input_ids)  # (seq_len, vocab)
            logits_pt = pt_logits_cache[i].to(ft_device)  # pre-computed, move to GPU

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
