import torch
from torch.nn import MultiheadAttention
from types import MethodType

from utils import rsetattr


class WrappedMultiheadAttention(torch.nn.Module):
    """
    Wrapped version of the MultiheadAttention Class
    - Call a MultiheadAttention module with "need_weights=True"
    - Store the attention weights using the given hook
    - Return the originaly asked result (with or without the weights)
    """
    def __init__(self, multihead_attention, name, attention_forward_hook):
        """
        __init__ function
        :param multihead_attention: The module to wrap
        :param name: The name to give to the hook function when storing the weights
        :param attention_forward_hook: The hook function to store the attention weights
        """
        super(WrappedMultiheadAttention, self).__init__()
        self.attn = multihead_attention
        self.name = name
        self.attention_forward_hook = attention_forward_hook

    def forward(self, query, key, value, **kwargs):
        # need_weight save the original behaviour to return or not the weights
        if "need_weight" not in kwargs:
            # Set to False if nothing is specified
            need_weight = False
        else:
            # Keep the asked behaviour is need_weight is specified
            need_weight = kwargs["need_weight"]

        # Set the kwargs need_weight argument for the inner MultiheadAttention to always True
        kwargs["need_weights"] = True
        # Get the result with the attention weights
        x, attention_weight_output = self.attn(query, key, value, **kwargs)
        # Store the attention weights using the hook function
        self.attention_forward_hook(self.name, attention_weight_output)

        # Return with or without the attention weights depending of the original need_weight parameter
        if need_weight:
            return x, attention_weight_output
        else:
            return x, None


def wrap_transformer(model):
    model.attention_weights = {}

    def set_attention_weights(self, layer_name, attention_weights):
        self.attention_weights[layer_name] = attention_weights

    model.set_attention_weights = MethodType(set_attention_weights, model)

    for name, module in list(model.named_modules()):
        if isinstance(module, MultiheadAttention):
            model.attention_weights[name] = None
            new_module = WrappedMultiheadAttention(module, name, model.set_attention_weights)
            rsetattr(model, name, new_module)