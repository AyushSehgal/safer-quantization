import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''



class OmniLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        # self.register_buffer('weight',ori_layer_norm.weight)
        self.register_parameter('weight', ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            # self.register_buffer('bias',ori_layer_norm.bias)
            self.register_parameter('bias', ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False
        # self.use_weight_quant = False


    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias

        # assert (weight.to(self.weight.device) != self.weight).sum() == 0, 'break'

        # torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, eps=self.eps)
        out = self.norm_func(x,self.normalized_shape,weight, bias,eps=self.eps)

        # if torch.isnan(out).sum() != 0 or torch.isinf(out).sum() != 0:

        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class QuantGemma3Norm(nn.Module):
    def __init__(self, ori_norm):
        super().__init__()
        self.eps = ori_norm.eps
        self.register_buffer('weight', ori_norm.weight)
        self.use_temporary_parameter = None
        self.eps1 = torch.ones(ori_norm.weight.shape, dtype=ori_norm.weight.dtype, device=ori_norm.weight.device, requires_grad=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())

        if self.use_temporary_parameter:
            weight = self.temp_weight
            eps1 = self.temp_eps1
        else:
            weight = self.weight
            eps1 = self.eps1

        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (eps1 + weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class OmniLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False
        self.use_weight_quant = False

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        return weight * hidden_states.to(input_dtype)

    # def forward(self, hidden_states):
    #
    #     input_dtype = hidden_states.dtype
    #     variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    #     hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    #
    #     if self.use_temporary_parameter:
    #         weight = self.temp_weight
    #         bias = self.temp_bias
    #     else:
    #         weight = self.weight
    #         bias = self.bias if hasattr(self, 'bias') else None
    #
    #     out = (weight * hidden_states+bias).to(input_dtype) if bias is not None else (weight * hidden_states).to(input_dtype)
    #
    #     return out


    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_weight_quant = use_weight_quant
        self.use_act_quant = use_act_quant