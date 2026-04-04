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
            inputs.pop("labels", None)

            print("\n" + "="*60)
            print("[DEBUG CAL] INJECTING GLOBAL FORWARD HOOKS TO TRACE NAN ORIGIN")
            
            nan_found = False
            def get_nan_hook(name):
                def hook(module, input, output):
                    nonlocal nan_found
                    if nan_found: return
                    
                    # Unpack output tensors
                    if isinstance(output, tuple):
                        out_tensors = [(idx, out) for idx, out in enumerate(output) if isinstance(out, torch.Tensor)]
                    elif isinstance(output, torch.Tensor):
                        out_tensors = [(0, output)]
                    else:
                        return
                        
                    for idx, tensor in out_tensors:
                        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                            # Check if the inputs were already corrupted
                            input_corrupted = False
                            if isinstance(input, tuple):
                                for inp in input:
                                    if isinstance(inp, torch.Tensor) and (torch.isnan(inp).any() or torch.isinf(inp).any()):
                                        input_corrupted = True
                            
                            # If inputs were clean but output is corrupted, THIS is the broken layer
                            if not input_corrupted:
                                issue_type = "NaN" if torch.isnan(tensor).any() else "Inf"
                                print(f"\n{'='*60}")
                                print(f"[FATAL BUG FOUND] {issue_type} generated exactly at module:")
                                print(f" -> Name: {name}")
                                print(f" -> Type: {type(module).__name__}")
                                print(f" -> Output Index: {idx}")
                                print(f"The inputs to this layer were clean, but it output {issue_type}s.")
                                print(f"{'='*60}\n")
                                nan_found = True
                                import sys; sys.exit(1)
                return hook

            # Register hooks on ALL modules in the fine-tuned model
            hooks = []
            for name, module in model.named_modules():
                # We skip the loss modules and focus entirely on the model graph
                hooks.append(module.register_forward_hook(get_nan_hook(name)))

            print("[DEBUG CAL] Running forward pass to catch the exact failing layer...")
            try:
                # This will trigger the hooks during the forward pass
                ft_outputs = self.get_ori_outputs(model, inputs)
            except Exception as e:
                print(f"[DEBUG CAL] Exception during forward pass: {e}")
            
            if not nan_found:
                print("[DEBUG CAL] Unquantized pass clean. Checking quantized pass...")
                try:
                    outputs = model(**inputs)
                except Exception as e:
                    pass

            # Cleanup
            for h in hooks: h.remove()
            if not nan_found:
                print("[DEBUG CAL] Hook trace complete. No NaNs found inside the layers?!")
                
            import sys; sys.exit(1)

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
