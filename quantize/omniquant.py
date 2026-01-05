import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_gemma3_layer import QuantGemma3DecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters, \
    omni_state_dict, register_scales_and_zeros, smooth_and_quant_temporary_raw, \
    smooth_and_quant_inplace_raw, clear_temp_variable, set_quant_state, smooth_and_quant_temporary
# from quantize.utils import other_parameters
import numpy as np
import torch.nn.functional as F
import functools
from tqdm import tqdm
from transformers.masking_utils import create_causal_mask

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)


def update_scales(model, train_dataloader, num_samples=128):

    model.eval()
    device = next(model.parameters()).device
    act_scales = {}
    calibration_dataset = train_dataloader

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, QuantLinear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    model = model.cuda()
    with torch.no_grad():
        # for i in tqdm(range(min(num_samples, len(calibration_dataset)))):
        for i, data in tqdm(enumerate(calibration_dataset)):
            tmp_data = {k: v.unsqueeze(0).cuda() for k, v in data.items()}
            model(**tmp_data)
            if i > (num_samples // tmp_data['input_ids'].shape[0]):
                break
            print(f"Processed {i + 1}/{num_samples} samples for scale estimation.")

    for h in hooks:
        # h = h.cpu()
        h.remove()

    return act_scales


def omniquant(
        lm,
        args,
        dataloader,
        act_scales,
        act_shifts,
        logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    # use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower() or 'gemma-2' in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1"
        }
        layer_name_prefix = "model.layers"
        # model.model.rotary_emb = model.model.rotary_emb.to(dev)
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif 'qwen' in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            # "o_proj": "out",
            "up_proj": "fc1"
        }
        layer_name_prefix = "model.layers"
    elif "gemma-3" in args.net.lower():
        is_llama = True
        layers = model.model.language_model.layers
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(dev)
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to(dev)
        model.model.language_model.norm = model.model.language_model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)
        DecoderLayer = QuantGemma3DecoderLayer
        pairs = {
            "q_proj": "qkv",
            # "o_proj": "out",
            "up_proj": "fc1"
        }
        model.config = model.config.text_config
        layer_name_prefix = "layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True  # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")

    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.bfloat16
        traincast = torch.cuda.amp.autocast

    args.nsamples = min(len(dataloader), 200)
    harm_ratio = 0.7 # 0.7 for llama2-7b
    harm_sample = args.nsamples * harm_ratio
    benign_sample = args.nsamples - harm_sample
    tmp = []
    for i in range(len(dataloader)):
        if len(tmp) <= args.nsamples:
            if dataloader[i][1] == 1 and harm_sample > 0:
                tmp.append(dataloader[i])
                harm_sample -= 1
            if dataloader[i][1] == 0 and benign_sample > 0:
                tmp.append(dataloader[i])
                benign_sample -= 1
        else:
            break
    dataloader = tmp

    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    seq_length = [(dataloader[i][0]['attention_mask'].sum() - 1).item() for i in range(len(dataloader))]
    print(seq_length)

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.attention_type = getattr(module, 'attention_type', None)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch, label in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(**batch)
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower() or "gemma-2" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif 'qwen' in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif 'gemma-3' in args.net.lower():
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.cpu()
        model.model.language_model.norm = model.model.language_model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)  # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None  # take output of quantization model as input

    # attention_mask = cache["attention_mask"]
    #
    # if attention_mask is not None:
    #     attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1,
    #                                                  1) if args.deactive_amp else attention_mask.repeat(args.batch_size,
    #                                                                                                     1, 1, 1).float()
    # else:
    #     logger.info(
    #         "No attention mask caught from the first layer."
    #         " Seems that model's attention works without a mask."
    #     )
    #     attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if 'gemma-3' in args.net.lower():
        position_embeddings = model.model.language_model.rotary_emb_local(fp_inps[0].unsqueeze(0), position_ids)
    else:
        position_embeddings = model.model.rotary_emb(fp_inps[0].unsqueeze(0), position_ids)

    regressions = np.load('./cls_finetuned_model_{}.npy'.format(args.net.lower()), allow_pickle=True).item()
    for k, v in regressions.items():
        v['w'] = torch.tensor(v['w'], dtype=inps[0].dtype, device=inps[0].device)
        v['b'] = torch.tensor(v['b'], dtype=inps[0].dtype, device=inps[0].device)

    if args.resume:
        omni_parameters = torch.load(args.resume)
        print(f"Resume from {args.resume}")
    else:
        omni_parameters = {}

    def compute_loss(fp_inp, quant_inp, j):

        label = dataloader[j][1]
        seq_l = seq_length[j]

        if label == 0:
            loss1 = loss_func(fp_inp, quant_inp)
            return loss1, [loss1.item(), 0.0, 0.0]
        else:
            # return torch.tensor(0, device=quant_inp.device), [0, 0, 0.0]
            loss2 = F.softplus(-(quant_inp[:, seq_l, :] @ w_i + b_i))[0]
            return loss2, [0, loss2.item(), 0.0]

        # return loss1 + lambda_attack * loss2 + lambda_prox * loss3, [loss1.item(), loss2.item(), loss3.item()]

    for i in range(len(layers)):

        w_i = regressions[i]['w']
        b_i = regressions[i]['b']

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear) and not "gate" in name:  # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args, layer_idx=i)
        qlayer = qlayer.to(dev)


        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        # causal_mask = lm.model.model._update_causal_mask(
                        #     dataloader[j][0]['attention_mask'], fp_inps[j].unsqueeze(0)
                        # )
                        causal_mask = create_causal_mask(
                            config=model.config,
                            input_embeds=fp_inps[j].unsqueeze(0),
                            attention_mask=dataloader[j][0]['attention_mask'],
                            cache_position=position_ids.squeeze(0),
                            past_key_values=None,
                            position_ids=position_ids.squeeze(0),
                        )
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=causal_mask, position_embeddings=position_embeddings,
                               position_ids=position_ids)[0]
                        # fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), position_embeddings_global, position_embeddings_local, position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=causal_mask,
                                                  position_ids=position_ids)[0]

        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True
        if 'llama' in args.net.lower() or args.abits == 16:
            use_shift = False  # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            if 'llama' in args.net.lower():
                qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(
                    torch.ones(layer.self_attn.q_proj.out_features, device=dev, dtype=dtype)))
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            # act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(
                            #     min=1e-5)
                            act = torch.rand(module.weight.shape[1], device=dev, dtype=dtype)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))

        if args.resume and i in omni_parameters:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()  # required for AMP training
            for param in qlayer.parameters():
                param.requires_grad = True
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                 {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}], weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            num_epoch = args.epochs - 5 if i <= len(layers) // 2 else args.epochs + 5
            # num_epoch = 0 if i in omni_parameters else num_epoch
            for epochs in range(num_epoch):
                loss_list = []
                loss_reconstruct = []
                loss_attack = []
                norm_list = []
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    # obtain output of quantization model
                    if dataloader[j][1] != 0:
                        continue
                    with traincast():
                        smooth_and_quant_temporary_raw(qlayer, args, is_llama)
                        # causal_mask = lm.model.model._update_causal_mask(
                        #     dataloader[j][0]['attention_mask'], fp_inps[j].unsqueeze(0)
                        # )
                        causal_mask = create_causal_mask(
                            config=model.config,
                            input_embeds=fp_inps[j].unsqueeze(0),
                            attention_mask=dataloader[j][0]['attention_mask'],
                            cache_position=position_ids.squeeze(0),
                            past_key_values=None,
                            position_ids=position_ids.squeeze(0),
                        )
                        quant_out = qlayer(quant_inps[index:index + args.batch_size, ], attention_mask=causal_mask, position_embeddings=position_embeddings,
                               position_ids=position_ids)[0]
                        loss, loss_l = compute_loss(fp_inps[index:index + args.batch_size, ].detach(), quant_out.unsqueeze(0), j)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index + args.batch_size, ], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()

                    loss_list.append(loss.detach().cpu())
                    loss_reconstruct.append(loss_l[0])
                    loss_attack.append(loss_l[1])

                    optimizer.zero_grad()

                    norm = loss_scaler(loss, optimizer, parameters=get_omni_parameters(qlayer, use_shift)).cpu()

                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                loss_reconstruct_mean = np.mean(loss_reconstruct)
                loss_attack_mean = np.mean(loss_attack)

                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} reconstruction:{loss_reconstruct_mean} attack:{loss_attack_mean}")

            clear_temp_variable(qlayer)
            del optimizer

        qlayer = qlayer.to(torch.bfloat16)
        # real smooth and quantization
        smooth_and_quant_inplace_raw(qlayer, args, is_llama)
        if args.epochs > 0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                # with traincast():
                    for j in range(args.nsamples):
                        # causal_mask = lm.model.model._update_causal_mask(
                        #     dataloader[j][0]['attention_mask'], fp_inps[j].unsqueeze(0)
                        # )
                        causal_mask = create_causal_mask(
                            config=model.config,
                            input_embeds=fp_inps[j].unsqueeze(0),
                            attention_mask=dataloader[j][0]['attention_mask'],
                            cache_position=position_ids.squeeze(0),
                            past_key_values=None,
                            position_ids=position_ids.squeeze(0),
                        )
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=causal_mask, position_embeddings=position_embeddings,
                               position_ids=position_ids)[0]
                        # quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=causal_mask,
                        #                        position_ids=position_ids, p=True)[0]

            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2, 3, 4] and args.abits >= 16  # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1)
                zeros = zeros.view(dim0, -1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features, module.out_features,
                                                        not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,
                                                          module.out_features, not module.bias is None)
                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)
                print(f"pack quantized {name} finished")
                del module
        del layer
        torch.cuda.empty_cache()


    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()

    return model

