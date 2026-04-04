import transformers, torch, os, datasets, random
import torch.nn.functional as F, torch, torch.nn as nn
import geoopt
from quant.cayley_opt import SGDG
from quant.ost_model_utils import SmoothModule, RotateModule
from transformers import AutoModelForCausalLM

import torch.distributed.fsdp as fsdp

fsdp.FullyShardedDataParallel


class MyTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        ):
            model: nn.Module = self.model
            ignored_modules = list()
            for m in model.modules():
                if isinstance(m, (RotateModule, SmoothModule)):
                    ignored_modules.append(m)
            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            self.accelerator.state.fsdp_plugin.use_orig_params = True

        if self.args.loss_type == "cal":
            assert self.args.pretrained_model, \
                "CAL loss requires --pretrained_model (path to unaligned base model)"
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(
                self.args.pretrained_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()
            for param in self.pretrained_model.parameters():
                param.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        args = self.args
        loss_type = args.loss_type
        if loss_type == "origin":
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if loss_type == "rkl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss
        if loss_type == "kl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        if (
            "r_kl_top" in loss_type
        ):  
            labels = inputs.pop("labels", None)
            if loss_type == "k_top":
                k = 1000
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        if "kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            if loss_type == "kl_top":
                k = 1000 
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if args.post_attn:
                ref = F.softmax(ori_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
            return (loss, outputs) if return_outputs else loss

        if loss_type == "mse":
            outputs = model(**inputs)
            logits = outputs["logits"]
            predict = torch.gather(logits, -1, indices.squeeze(1))
            loss = F.mse_loss(predict, values.squeeze(1))
            return (loss, outputs) if return_outputs else loss
        elif loss_type == "kl":
            outputs = model(**inputs)
            logits = outputs["logits"]
            predict = torch.gather(logits, -1, indices.squeeze(1))
            predict = F.softmax(predict, dim=-1)
            values = F.softmax(values.squeeze(1), dim=-1)

            predict_dist = torch.distributions.Categorical(probs=predict)
            values_dist = torch.distributions.Categorical(probs=values)

            loss = torch.distributions.kl_divergence(predict_dist, values_dist).mean()
            return (loss, outputs) if return_outputs else loss
        elif loss_type == "rkl":
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits

            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return (loss, outputs) if return_outputs else loss

        if loss_type == "cal":
            k = self.args.cal_top_k
            alpha = self.args.cal_alpha
            inputs.pop("labels", None)

            # 1. Forward passes
            ft_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            q_logits = outputs.logits
            pt_logits = self.get_pretrained_outputs(inputs).logits

            # --- DEBUG BLOCK 1: Forward Pass Integrity ---
            # If any of these trigger, the bug is in the model's weights or quant ops, not the loss.
            assert not torch.isnan(ft_logits).any(), "[DEBUG] NaNs found in ft_logits! The unquantized fine-tuned model forward pass failed."
            assert not torch.isnan(q_logits).any(), "[DEBUG] NaNs found in q_logits! The quantized model forward pass failed (likely a zero-division in quant scales)."
            assert not torch.isnan(pt_logits).any(), "[DEBUG] NaNs found in pt_logits! The pre-trained base model forward pass failed."
            
            assert not torch.isinf(ft_logits).any(), "[DEBUG] Infs found in ft_logits!"
            assert not torch.isinf(q_logits).any(), "[DEBUG] Infs found in q_logits!"
            assert not torch.isinf(pt_logits).any(), "[DEBUG] Infs found in pt_logits!"

            # Cast to float32
            ft_logits = ft_logits.float()
            q_logits = q_logits.float()
            pt_logits = pt_logits.to(ft_logits.device).float()

            with torch.no_grad():
                p_ft = F.softmax(ft_logits, dim=-1).clamp(min=1e-8)
                p_pt = F.softmax(pt_logits, dim=-1).clamp(min=1e-8)
                
                # --- DEBUG BLOCK 2: Probability Integrity ---
                assert not torch.isnan(p_ft).any(), "[DEBUG] NaNs found in p_ft after softmax."
                assert not torch.isnan(p_pt).any(), "[DEBUG] NaNs found in p_pt after softmax."

                log_p_ft = torch.log(p_ft)
                log_p_pt = torch.log(p_pt)

            _, s_top = p_ft.topk(k, dim=-1, sorted=False)
            _, s_diff = (p_ft - p_pt).abs().topk(k, dim=-1, sorted=False)

            log_q = F.log_softmax(q_logits, dim=-1).clamp(min=-20.0)
            
            # --- DEBUG BLOCK 3: Quantized Log-Prob Integrity ---
            assert not torch.isnan(log_q).any(), "[DEBUG] NaNs found in log_q after log_softmax."

            p_ft_stop = p_ft.gather(-1, s_top)
            log_p_ft_stop = log_p_ft.gather(-1, s_top)
            log_q_stop = log_q.gather(-1, s_top)

            p_pt_sdiff = p_pt.gather(-1, s_diff)
            log_p_pt_sdiff = log_p_pt.gather(-1, s_diff)
            log_q_sdiff = log_q.gather(-1, s_diff)

            l_kl_top = (p_ft_stop * (log_p_ft_stop - log_q_stop)).sum(dim=-1).mean()
            l_cont_top = (p_pt_sdiff * (log_p_pt_sdiff - log_q_sdiff)).sum(dim=-1).mean()

            # --- DEBUG BLOCK 4: Final Loss Integrity ---
            assert not torch.isnan(l_kl_top), f"[DEBUG] NaNs in l_kl_top! p_ft_stop range: {p_ft_stop.min().item()} to {p_ft_stop.max().item()}"
            assert not torch.isnan(l_cont_top), "[DEBUG] NaNs in l_cont_top!"

            loss = l_kl_top - alpha * l_cont_top
            
            assert not torch.isnan(loss), "[DEBUG] Total loss evaluated to NaN!"

            return (loss, outputs) if return_outputs else loss

        if loss_type == "mse":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.mse_loss(logits, ori_logits)
            return (loss, outputs) if return_outputs else loss
        if loss_type == "kd":
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            T, alpha = self.temperature, self.loss_alpha
            ori_loss = outputs["loss"]
            logits = logits.view(-1, logits.size(-1))
            ori_logits = ori_logits.view(-1, ori_logits.size(-1))
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
            return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        args = self.args
        inputs = dict(inputs)
        inputs.pop("labels", None)
        acc = self.accelerator

        
        def set_temporary(model, temporary=True):
            model.temporary = temporary
            model.model.temporary = temporary
            model.model.embed_tokens.temporary = temporary
            model.lm_head.temporary = temporary
            for layer in model.model.layers:
                layer.set_temporary(temporary)

        def set_quant_state(
            model,
            use_weight_quant: bool = False,
            use_act_quant: bool = False,
            use_fully_quant: bool = False,
        ):
            model.model.norm.use_act_quant = (
                use_fully_quant
            )
            model.model.embed_tokens.use_act_quant = (
                use_fully_quant
            )
            for layer in model.model.layers:
                layer.set_quant_state(use_weight_quant, use_act_quant, use_fully_quant)

        set_temporary(acc.unwrap_model(model), False)
        set_quant_state(acc.unwrap_model(model), False, False, False)
        outputs = model(**inputs, output_hidden_states=True)
        set_temporary(acc.unwrap_model(model), True)
        set_quant_state(
            acc.unwrap_model(model), args.train_enable_wquant, True, args.fully_quant
        )
        return outputs

    @torch.no_grad()
    def get_pretrained_outputs(self, inputs):
        inputs = {k: v for k, v in inputs.items() if k != "labels"}
        device = next(self.pretrained_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self.pretrained_model(**inputs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        
        
        args = self.args
        params_rotate = []
        params_smooth = []
        for param in self.model.parameters():
            param: torch.nn.Parameter
            if param.requires_grad:
                if len(param.size()) == 2:
                    params_rotate.append(param)
                else:
                    params_smooth.append(param)
        dict_rotate = {
            "params": params_rotate,
            "lr": args.rotate_lr,
            "momentum": args.rotate_momentom,
            "stiefel": True,
            "grassmann": True,
            "omega": 0.1,
        }
        dict_smooth = {
            "params": params_smooth,
            "lr": args.smooth_lr,
            "momentum": args.smooth_momentom,
            "stiefel": False,
            "nesterov": False,
        }
        if args.opt_type == "SGDG":
            optimizer = SGDG(
                [dict_rotate, dict_smooth], weight_decay=0
            )  
        elif args.opt_type == "RSGD":
            optimizer = geoopt.optim.RiemannianSGD(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10,
            )
        elif args.opt_type == "RAdam":
            optimizer = geoopt.optim.RiemannianAdam(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.rotate_lr,stabilize=10
            )
        self.optimizer = optimizer
        
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )
