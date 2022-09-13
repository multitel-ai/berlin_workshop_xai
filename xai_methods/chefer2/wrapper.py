import torch
from torch.nn import MultiheadAttention
from types import MethodType

from utils import rsetattr


class WrappedMultiheadAttention(torch.nn.Module):
    """
    Wrapped version of the MultiheadAttention Class
    - Call a MultiheadAttention module with "need_weights=True"
    - Store the attention weights using the given hook
    - Return the originally asked result (with or without the weights)
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
        # need_weight save the original behaviour to return or not the attention weights
        if "need_weight" not in kwargs:
            # Set to False if nothing is specified
            need_weight = False
        else:
            # Store the originally asked behaviour is need_weight is specified
            need_weight = kwargs["need_weight"]

        # Set need_weight argument in the kwargs to always True for the inner MultiheadAttention
        kwargs["need_weights"] = True
        # Get the result with the attention weights
        x, attention_weight_output = self.attn(query, key, value, **kwargs)
        # Store the attention weights using the hook function
        self.attention_forward_hook(self.name, attention_weight_output)

        # Return the result with or without the attention weights depending on the original need_weight parameter
        if need_weight:
            return x, attention_weight_output
        else:
            return x, None


def wrap_transformer(model):
    """
    The function wrap a transformer to allow easy retrieval of the attention weights
    - Add a hook method that store the weights in a self.attention_weights dictionary, named by the modules names
    - Wrap every MultiheadAttention module to get weights and give them to the hook function
    Warning: This function change the model in-place, it does not create a copy
    :param model: Transformer module to wrap
    :return: The wrapped module (optional since the model is modified in-place)
    """
    # Create the dictionary to store the weights as an attribute of the model
    model.attention_weights = {}

    # define the hook function
    def set_attention_weights(self, layer_name, attention_weights):
        self.attention_weights[layer_name] = attention_weights

    # Add the hook function as a method to the model
    model.set_attention_weights = MethodType(set_attention_weights, model)

    # Loop over the modules to find and wrap the MultiheadAttention modules
    for name, module in list(model.named_modules()):
        if isinstance(module, MultiheadAttention):
            # Create the named key in the dictionary
            model.attention_weights[name] = None
            # Wrap the current MultiheadAttention module
            new_module = WrappedMultiheadAttention(module, name, model.set_attention_weights)
            # Use the new wrapped module
            rsetattr(model, name, new_module)

    return model
