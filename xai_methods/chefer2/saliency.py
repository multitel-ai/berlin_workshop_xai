import torch

# def salience(input1, input2, model, device, start_layer=-1):
#   '''
#   This method calculates the salience value using the method proposed in Hila et al, 2021 see here: https://arxiv.org/pdf/2103.15679.pdf
#   input1: image or text
#   input2: image or text
#   model: CLIP model to generate the salience of the attention layers with respect to the output
#   device: cpu or gpu
#   start_layer: layers to start the salience computation. default is last layer.
#   '''

#   def forward_backward(model, input1, input2):
#     '''
#     This method does forward and backward propagation to get the attention weights and gradients of the model.
#     '''
#     # Forward propagation to calculate the model output along with attentions.
#     out1, out2 = model(input1, input2)

#     # Backward propagation to compute the gradients with respective to model output from previos step.
#     out1.backward(retain_graph=True)

#   def relevance_score(model, batch_size, start_layer):
#     '''
#     This method calculates the relevence score considered as the salience of the inputs using method proposed by Hile et al, 2021.
#     '''

#     # Coverting attention weight hook dictionary keys into a list
#     keys = list(model.attention_weights.keys())

#     if start_layer == -1:
#       start_layer = len(keys) - 1

#     num_tokens = model.attention_weights[keys[0]].shape[-1]
#     R = torch.eye(num_tokens, num_tokens, dtype=model.attention_weights[keys[0]].dtype).to(device)
#     R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

#     for i,key in enumerate(keys):

#       if i < start_layer:
#         continue

#       grad = model.attention_grads[key].detach()
#       cam = model.attention_weights[key].detach()
#       cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#       grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#       cam = grad * cam
#       cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
#       cam = cam.clamp(min=0).mean(dim=1)
#       R = R + torch.bmm(cam, R)

#     return R

#   forward_backward(model, input1, input2)

#   batch_size = input1.shape[0]

#   input1_relevance = relevance_score(model.visual, batch_size, start_layer)[:, 0, 1:]
#   CLS_idx = input2.argmax(dim=-1)
#   input2_relevance = relevance_score(model.transformer, batch_size, start_layer)[:,CLS_idx, 1:CLS_idx]


#   return input1_relevance, input2_relevance

def forward_prop(input, model):
  '''
  This method does the forward propagation on the CLIP model either on the visual part or text part based on the input.
  input: image or text tensors
  model: CLIP model
  '''

  if len(input.shape) > 2: # encode method of the CLIP model is called based on the shape of the input
    feature_out = model.encode_image(input)
  else:
    feature_out = model.encode_text(input)

  return feature_out


def cosine_similarity(input_1, input_2=None, model=None):
  '''
  This method calculates the cosine similarity between inputs and returns the similarity value to be used for backward propagation.
  input_1: image or text
  input_2: image or text
  model: CLIP model
  '''

  if input_2 != None:

    feature_1 = forward_prop(input_1, model) # forward propagation to get the final encoder layer output.
    feature_2 = forward_prop(input_2, model) # forward propagation to get the final encoder layer output.

    # normalized features
    feature_1 = feature_1 / feature_1.norm(dim=1, keepdim=True)
    feature_2 = feature_2 / feature_2.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_feat_1 = logit_scale * feature_1 @ feature_2.t()

  else:

    features = forward_prop(input_1, model) # forward propagation to get the final encoder layer output.

    # normalized features
    features = features / features.norm(dim=1, keepdim=True)

    # selecting indexes
    ind_1 = [i for i in range(0, input_1.shape[0], 2)]
    ind_2 = [i for i in range(1, input_1.shape[0], 2)]

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_feat_1 = logit_scale * features[ind_1] @ features[ind_2].t()

  logits_per_feat_2 = logits_per_feat_1.t()

  # shape = [global_batch_size, global_batch_size]
  return logits_per_feat_1, logits_per_feat_2


def salience_modular(input_1, input_2=None, model=None, device='cpu', start_layer=-1):
  '''
  This method calculates the salience value using the method proposed in Hila et al, 2021 see here: https://arxiv.org/pdf/2103.15679.pdf
  input1: image or text
  input2: image or text
  model: CLIP model to generate the salience of the attention layers with respect to the output
  device: cpu or gpu
  start_layer: layers to start the salience computation. default is last layer
  '''

  # forward propagation with cosine similarity calculation
  cosine_input_1, cosine_input_2 = cosine_similarity(input_1, input_2, model)

  # backward propagation to populate the gradients in the network
  cosine_input_1.backward(retain_graph=True)

  # salience calculation
  if input_2 != None:
    relevance_input_1 = relevance_score(input_1, model, device, start_layer)
    relevance_input_2 = relevance_score(input_2, model, device, start_layer)

  else:
    relevance_input = relevance_score(input_1, model, device, start_layer)

    # selecting indexes to split the data when two image or two text compared.
    ind_1 = [i for i in range(0, input_1.shape[0], 2)]
    ind_2 = [i for i in range(1, input_1.shape[0], 2)]

    relevance_input_1 = relevance_input[ind_1]
    relevance_input_2 = relevance_input[ind_2]

  return relevance_input_1, relevance_input_2


def relevance_score(input, model, device, start_layer):
  '''
  This method calculates the relevence score considered as the salience of the inputs using method proposed by Hile et al, 2021.
  '''

  if len(input.shape) > 2:
    model = model.visual
    visual = True
  else:
    visual = False
    model = model.transformer

  batch_size = input.shape[0]

  keys = list(model.attention_weights.keys())  # obtaining the keys of each attention layer.

  if start_layer == -1:
    start_layer = len(keys) - 1

  num_tokens = model.attention_weights[keys[0]].shape[-1]
  R = torch.eye(num_tokens, num_tokens, dtype=model.attention_weights[keys[0]].dtype).to(device)
  R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

  for i, key in enumerate(keys):

    if i < start_layer:
      continue

    grad = model.attention_grads[key].detach()
    cam = model.attention_weights[key].detach()
    cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
    cam = grad * cam
    cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
    cam = cam.clamp(min=0).mean(dim=1)
    R = R + torch.bmm(cam, R)

  if visual:
    R = R[:, 0, 1:]
  else:
    # CLS_idx = input.argmax(dim=-1)
    # R = R[:,CLS_idx, 1:CLS_idx]
    pass

  return R
