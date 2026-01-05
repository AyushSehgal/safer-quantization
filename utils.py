import torch
from models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTAttention
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_gemma3_layer import QuantGemma3DecoderLayer
from quantize.int_linear import QuantLinear
from quantize.utils import smooth_and_quant_temporary, smooth_and_quant_inplace
from datautils import get_loaders
from tqdm import tqdm
import torch.nn as nn
import os
import re

# from torch._six import inf
from math import inf
import logging
from termcolor import colored
import sys
import time

def model_quantization(model, model_name, w_train_bits, a_train_bits, resume=None):

    model_nick_name = model_name.split("/")[-1]
    is_llama = 'opt' not in model_name.lower()

    act_scales = torch.load(f'./act_scales/{model_nick_name}.pt')
    if not is_llama:
        act_shifts = torch.load(f'./act_shifts/{model_nick_name}.pt')


    quant_args = {"weight_quant_params": {'n_bits': w_train_bits, 'per_channel_axes': [0], 'symmetric': False,
                                          'dynamic_method': 'per_channel', 'group_size': False, 'lwc': True,
                                          'disable_zero_point': False},
                  "act_quant_params": {'n_bits': a_train_bits, 'per_channel_axes': [], 'symmetric': False,
                                       'dynamic_method': 'per_token'},
                  "p_quant_params": {'n_bits': 16, 'metric': 'fix0to1'}}


    if 'opt' in model_name.lower():
        layer_name_prefix = "model.decoder.layers"
        layers = model.model.decoder.layers
        Qlayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
    elif 'llama' in model_name.lower():
        layer_name_prefix = "model.layers"
        layers = model.model.layers
        Qlayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1"
        }
    elif 'gemma' in model_name.lower():
        layer_name_prefix = "layers"
        layers = model.model.language_model.layers
        Qlayer = QuantGemma3DecoderLayer
        pairs = {
            "q_proj": "qkv",
            # "o_proj": "out",
            "up_proj": "fc1"
        }
    elif 'qwen' in model_name.lower():
        layer_name_prefix = "model.layers"
        layers = model.model.layers
        Qlayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "up_proj": "fc1"
        }

    model.config.weight_quant_params = quant_args["weight_quant_params"]
    model.config.act_quant_params = quant_args["act_quant_params"]
    model.config.p_quant_params = quant_args["p_quant_params"]

    alpha = 0.5
    qlinears = []
    for i in range(len(layers)):
        layer = layers[i]
        qlayer = Qlayer(config=model.config, ori_layer=layer, args=model.config, layer_idx=i)
        qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(
            torch.ones(layer.self_attn.q_proj.out_features, device=layer.self_attn.q_proj.weight.device,
                       dtype=torch.bfloat16)))
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                qlinears.append(module)
                for key in pairs.keys():
                    if key in name:
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                        act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=weight.device,
                                                                                         dtype=torch.bfloat16).clamp(
                            min=1e-5)
                        # scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                        if 'llama' in model_name.lower():
                            min1 = 0.75
                            max1 = 0.75
                            refer = act / weight
                            alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                            scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                        elif 'opt' in model_name.lower():
                            if key == 'q_proj':
                                # max1 = 0.75
                                # min1 = 0.75 if i != 0 else 0.7
                                max1 = 0.7
                                min1 = 0.6
                                # llama是0.5到0.7
                                refer = act / weight
                                alpha = min1 + (refer - refer.min()) / (refer.max() - refer.min()) * (max1 - min1)
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)
                            else:
                                scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)

                        elif 'gemma' in model_name.lower():
                            scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)

                        elif 'qwen' in model_name.lower():
                            scale = (act.pow(alpha) / weight.pow(1 - alpha)).clamp(min=1e-5)

                        shift = torch.zeros_like(scale, device=weight.device, dtype=torch.bfloat16)

                        # if key == 'o_proj' or key == 'out_proj':
                        #     qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(
                        #         torch.zeros(scale.shape, device=layer.self_attn.q_proj.weight.device,
                        #                     dtype=torch.bfloat16)))
                        #     qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(
                        #         torch.ones(scale.shape, device=layer.self_attn.q_proj.weight.device,
                        #                    dtype=torch.bfloat16)))
                        # else:
                        qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                        qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

        layers[i] = qlayer


    for param in qlayer.parameters():
        param.requires_grad = False


    resume = './log/Llama-2-7b-chat-alpaca-hr0-w8a8/omni_parameters.pth' if resume is None else resume
    omni_parameters = torch.load(resume)
    for i in range(len(layers)):
        if resume and i in omni_parameters:
            layers[i].load_state_dict(omni_parameters[i], strict=False)

    idx = 0
    for layer in layers:
        with torch.no_grad():
            smooth_and_quant_temporary(layer, a_train_bits, model_name)
            # smooth_and_quant_inplace(layer, a_train_bits, model_name)
        idx += 1

    if a_train_bits < 16:
        for linear in qlinears:
            linear.use_act_quant = True


    return model, qlinears

@torch.no_grad()
def evaluate(model, model_name):
    results = {}

    model.seqlen = 2048
    cache_dir = './cache'

    model_family = re.findall(r"/(.*?)-", model_name)[0]

    for dataset in ["wikitext2"]:
        cache_testloader = f'{cache_dir}/testloader_{model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            dataloader, testloader = get_loaders(
                dataset,
                seed=42,
                model=model_name,
                seqlen=model.seqlen,
            )
            torch.save(testloader, cache_testloader)
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // model.seqlen
        model.config.use_cache = False
        model.eval()
        nlls = []
        model = model.to('cuda')
        model = model.to(torch.bfloat16)
        total_nll = 0.0
        total_tokens = 0
        with torch.inference_mode():
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(model.device)
                attention_mask = torch.ones(batch.shape, dtype=torch.long, device=batch.device)
                with torch.no_grad():
                    outputs = model.model(batch, attention_mask=attention_mask)
                hidden_states = outputs[0]
                logits = model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)][
                               :, 1:
                               ].to(model.lm_head.weight.device)

                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_nll += loss.item()
                total_tokens += shift_labels.numel()

                print(loss.item())


        torch.cuda.empty_cache()
        # ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * model.seqlen))
        ppl = torch.exp(torch.tensor(total_nll / total_tokens))
        print(f'{dataset} : {ppl.item()}')
        model.config.use_cache = False
        results[dataset] = ppl.item()

    return results

@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger