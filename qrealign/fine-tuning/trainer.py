from torch.cuda.amp import autocast
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from transformers import Trainer
from transformers import logging

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
# from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from torch.cuda.amp import autocast, GradScaler
from transformers.utils import is_sagemaker_mp_enabled


def get_leaf_modules_with_grad(module):
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention):
        # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list

class BoosterAlignmentTrainer(Trainer):

    def get_harmful_dataloader(self, harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator

        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))

    def init(self, harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0

    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            # print(grad_output[0])

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list

        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data = output[0] + perturbation
            # print(output.shape)
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([
                # original sam
                (poison_grads_representation[name]).norm(p=2)
                # asam
                # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            # first backward gradient for harmful dataset
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if
                            param.requires_grad}
            model.zero_grad()

            # Take step with the harmful perturbation
            with torch.no_grad():
                grad_norm = self._grad_norm(stored_grads) + 1e-7
            # perturb the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data += self.args.rho*stored_grads[name]/grad_norm
                    param.data -= self.args.alpha * stored_grads[name] / grad_norm

            # backward the gradient after harmful perturbation
            with self.compute_loss_context_manager():
                loss2 = self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            perturb_grads = {name: param.grad.clone() for name, param in model.named_parameters() if
                             param.requires_grad}

            model.zero_grad()

            # recover the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += self.args.alpha * stored_grads[name] / grad_norm

            if self.args.perturb_aware == "True":
                self.sam_state = {}
                self.sam_state["hooks"] = []
                self.sam_state["gradient"] = {}
                # do forward backward on safety data
                self.pre_first_step(model)
                # first backward
                loss4 = self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss4, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss4)
                self.after_first_step(model)
                model.zero_grad()
                self.pre_second_step(model)
                loss3 = self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                # cancel the perturbation
                self.after_second_step(model)
                # sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # param.grad.data=param.grad.data - (self.args.alpha +self.args.lamb/self.args.rho)*stored_grads[name] +self.args.lamb/self.args.rho* perturb_grads[name]
                        param.grad.data = param.grad.data + (self.args.lamb) * stored_grads[name] - self.args.lamb * \
                                          perturb_grads[name]

                self.steps += 1
                if self.steps % 500 == 0:
                    self.statistic = 0
                    self.statistic += sum(
                        [torch.norm(stored_grads[name]) ** 2 for name, param in model.named_parameters() if
                         param.requires_grad]).detach()
                    print("harmful gradient norm {}".format(self.statistic), flush=True)
                    print("harmful loss {}".format(loss), flush=True)
                return loss3
            else:
                # else:
                # Finally backward for minimize safety gradient
                # print(loss)
                # calculate the alignment grad
                with self.compute_loss_context_manager():
                    loss3 = self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)

                # Finally, sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data = param.grad.data + (self.args.lamb) * stored_grads[name] - self.args.lamb * \
                                          perturb_grads[name]

                self.steps += 1
                if self.steps % 2000 == 0:
                    self.statistic = 0
                    self.statistic += grad_norm.detach()
                    # self.statistic += loss-loss2
                    print("harmful gradient norm {}".format(self.statistic), flush=True)
                    print("loss change {}".format(loss - loss2), flush=True)
                    print("harmful loss {}".format(loss), flush=True)
            return loss3

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps


class Panacea(BoosterAlignmentTrainer):
    def init(self, harmful_datast, model, tag):
        self.clock = 0
        self.steps = 0
        # if self.args.guide_data_num > 0:
        self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
        self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0
        self.tag = tag
        self.eval_metric = []
        if self.tag == "eps" or self.tag == "gw" or self.tag == "log":
            self.epsilon = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.epsilon[name] = torch.zeros_like(param.data, requires_grad=False)
        self.scaler = GradScaler()

    def step_eps(self, model, inputs, harmful_inputs):

        with autocast(dtype=torch.bfloat16):
            with self.compute_loss_context_manager():
                loss_g = self.compute_loss(model, inputs)

        self.scaler.scale(loss_g).backward()

        stored_g_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        model.zero_grad()

        # optimize h(w)
        with autocast(dtype=torch.bfloat16):
            with self.compute_loss_context_manager():
                loss_h_origin = self.compute_loss(model, harmful_inputs)

        self.scaler.scale(loss_h_origin).backward()

        stored_h_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # compute epsilon
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.epsilon[name] = p.grad.detach().clone()

        epsilon_norm = self._grad_norm(self.epsilon)

        # perturb w -> w + eps
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.epsilon[name] *= self.args.eps_rho / (epsilon_norm + 1e-7)
                p.data.add_(self.epsilon[name])

        model.zero_grad()

        # compute h(w + eps)
        with autocast(dtype=torch.bfloat16):
            with self.compute_loss_context_manager():
                loss_h = self.compute_loss(model, harmful_inputs)
                loss_h = torch.clamp(loss_h, max=5)

        self.scaler.scale(loss_h).backward()

        # restore w -> w
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.sub_(self.epsilon[name])

        # combine gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.grad.data = -(
                        self.args.lamb * (p.grad.data - stored_h_grads[name])
                        - stored_g_grads[name]
                )

        # update scaler + optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
        model.zero_grad()

        # logging
        self.steps += 1
        if (self.steps - 1) % 500 == 0:
            self.statistic = epsilon_norm

        return loss_h

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch
    ) -> torch.Tensor:

        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        loss = self.step_eps(model, inputs, harmful_inputs)
        return loss.detach() / self.args.gradient_accumulation_steps

    def get_epsilon(self):
        return self.epsilon


class LisaTrainer(Trainer):

    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator

        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))

    def init(self, alignment_dataset):
        if self.args.alignment_step != 0 and self.args.guide_data_num > 0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        self.alignment_weights = {}
        self.finetune_weights = {}
        # self.gamma ={}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)

    def end_training(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.status == "alignment":
                    self.alignment_weights[name] = param.data.detach().clone()
                else:
                    self.finetune_weights[name] = param.data.detach().clone()

    def switch_model(self):
        sum_drift = 0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("finetuning drift to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("alignment drift to consensus{}".format(sum_drift))

    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    def check_mode(self, inputs):
        if self.status == "alignment":
            if self.clock % (self.args.alignment_step) == 0 and self.steps != 0 and self.args.finetune_step != 0:
                self.status = "finetune"
                self.switch_model()
                self.clock = 0

            else:
                inputs = self.sample_from_alignment()
        else:
            if self.clock % (self.args.finetune_step) == 0 and self.steps != 0 and self.args.alignment_step != 0 and self.args.guide_data_num > 0:
                self.status = "alignment"
                self.switch_model()
                # alignment need another input

                inputs = self.sample_from_alignment()
                self.clock = 0
        return inputs

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch
    ) -> torch.Tensor:
        # may change input due to mode change
        inputs = self.check_mode(inputs)
        model.train()

        inputs = self._prepare_inputs(inputs)

        def step():

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.status == "alignment":
                # print("alignment_loss_prev: {}".format(loss.item()))
                if self.steps > 0.1 * len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        if param.requires_grad and self.args.rho > 0:
                            # loss +=torch.sum(self.gamma[name] *  param)+ self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                            loss += self.args.rho / 2 * torch.norm(param - self.finetune_weights[name]) ** 2
                # print("alignment_loss: {}".format(loss.item()))
            else:
                # print("finetune_loss_prev: {}".format(loss.item()))

                if self.steps > 0.1 * len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        # we observe that for Gsm8k, proximal term will hurt convergence. Don't do proximal for the first few rounds.
                        if param.requires_grad and self.args.rho > 0:
                            # loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                            loss += self.args.rho / 2 * torch.norm(param - self.alignment_weights[name]) ** 2

            self.accelerator.backward(loss)
            return loss

        loss = step()
        self.steps += 1
        self.clock += 1
        return loss.detach() / self.args.gradient_accumulation_steps