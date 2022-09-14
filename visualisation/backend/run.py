import io
import json

import torch
from PIL import Image
from flask import Flask, jsonify, request
import requests

import numpy as np

from CLIP.clip import clip

from xai_methods.chefer2 import wrap_transformer
from xai_methods.chefer2 import chefer2_saliency

app = Flask(__name__)

# Define the model for clip
model_type = "ViT-B/32"
# The patch size must correspond to the model size
patch_size = 32

# Load the model and send it to gpu if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, preprocess = clip.load(model_type)
model.to(device)

# Wrap the visual and text parts of clip to get the attentions weights
visual_transformer = wrap_transformer(model.visual)
text_transformer = wrap_transformer(model.transformer)

# Requests sent to the /attentions endpoints return the attention weights
@app.route('/attentions', methods=['POST'])
def attentions():

    # Check if 'Content-Type is json
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = json.loads(request.data)
    else:
        return 'Content-Type not supported!'

    # Download the image from the given url and convert it to a Image class
    image_url = data['img_url']
    r = requests.get(image_url)
    image = Image.open(io.BytesIO(r.content))

    # Get the text
    text = data['text']

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    # Tokenize the text
    tokens = clip.tokenize(text).to(device)

    # Preprocess the image into a tensor with the right dimensions
    image = preprocess(image).to(device).unsqueeze(0)

    # Compute the dimensions of the image tokens based on the image size and patch size
    image_tokens_size = torch.tensor(image.shape[-2:]) // patch_size

    # Compute the coordinates of the image tokens
    y_size = image_tokens_size[0].item()
    x_size = image_tokens_size[1].item()
    y = torch.tensor(range(y_size)).unsqueeze(1)
    x = torch.tensor(range(x_size)).unsqueeze(0)
    coord_tensor = torch.stack((y.repeat((1, y_size)), x.repeat((x_size, 1)))) * patch_size
    img_coords_list = coord_tensor.flatten(1).T.numpy().tolist()


    img_coords = [(subList[0], subList[1]) for subList in img_coords_list]

    # Compute the embeddings for the image and the text
    image_embedding = visual_transformer(image.type(model.dtype))
    text_embedding = model.encode_text(tokens)

    #TODO hook the embedding at each attention layer in wrapper?

    # Compute the cos similarity between the embeddings
    cos_sim = image_embedding @ text_embedding.t()

    # Retrieve the attentions weights for image and text
    image_attentions = visual_transformer.attention_weights
    text_attentions = text_transformer.attention_weights

    # cat the attentions weights dict to a tensor (n_layers, n_attention_heads_per_layer, n_tokens, n_tokens)
    image_attentions = torch.cat([attn[1].detach().cpu() for attn in list(image_attentions.items())])
    text_attentions = torch.cat([attn[1].detach().cpu() for attn in list(text_attentions.items())])

    text_tokens = ["<CLS>"] + text[0].strip().split(" ") + ["<SEP>"]

    #image_relevance, text_relevance = chefer2_saliency(image, tokens, model, device)
    image_relevance, text_relevance = 0, 0

    # Send the response in a json
    response = jsonify({'img_emb': image_embedding.detach().cpu().numpy().tolist(),
                        'txt_emb': text_embedding.detach().cpu().numpy().tolist(),
                        'image_attention': image_attentions.numpy().tolist(),
                        'text_attention': text_attentions.numpy().tolist(),
                        'img_coords': img_coords,
                        'cos_sim': cos_sim.detach().cpu().item(),
                        'image_relevance': image_relevance,
                        'text_relevance': text_relevance,
                        'tokens': text_tokens
                        })

    return response


if __name__ == "__main__":

    app.run(host="0.0.0.0")
