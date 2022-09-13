import io
import json

import torch
from PIL import Image
from flask import Flask, jsonify, request
import requests

import numpy as np

from CLIP.clip import clip

from xai_methods.chefer2 import wrap_transformer

app = Flask(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, preprocess = clip.load("ViT-B/32")
model.to(device)

@app.route('/attentions', methods=['POST'])
def attentions():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = json.loads(request.data)
    else:
        return 'Content-Type not supported!'

    image_url = data['img_url']
    r = requests.get(image_url)
    image = Image.open(io.BytesIO(r.content))
    text = data['text']

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    visual_transformer = wrap_transformer(model.visual)
    text_transformer = wrap_transformer(model.transformer)

    tokens = clip.tokenize(text).to(device)
    image = preprocess(image).to(device).unsqueeze(0)

    image_embedding = visual_transformer(image.type(model.dtype))
    text_embedding = model.encode_text(tokens)

    image_attentions = visual_transformer.attention_weights
    text_attentions = text_transformer.attention_weights

    # cat the attentions weights dict to a tensor (n_layers, n_attention_heads_per_layer, n_tokens, n_tokens)
    image_attentions = torch.cat([attn[1].detach().cpu() for attn in list(image_attentions.items())])
    text_attentions = torch.cat([attn[1].detach().cpu() for attn in list(text_attentions.items())])


    response = jsonify({'img_emb': image_embedding.detach().cpu().numpy().tolist(),
                'txt_emb': text_embedding.detach().cpu().numpy().tolist(),
                'image_attention': image_attentions.numpy().tolist(),
                'text_attention': text_attentions.numpy().tolist(),
                })

    return response


if __name__ == "__main__":

    app.run(host="0.0.0.0")
