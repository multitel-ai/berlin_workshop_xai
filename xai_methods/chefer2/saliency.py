def salience(input1, input2, model, device, start_layer=-1):
  '''
  This method calculates the salience value using the method proposed in Hila et al, 2021 see here: https://arxiv.org/pdf/2103.15679.pdf
  input1: image or text
  input2: image or text
  model: CLIP model to generate the salience of the attention layers with respect to the output
  device: cpu or gpu
  start_layer: layers to start the salience computation. default is last layer.
  '''

  def forward_backward(model, input1, input2):
    '''
    This method does forward and backward propagation to get the attention weights and gradients of the model.
    '''
    # Forward propagation to calculate the model output along with attentions.
    out1, out2 = model(input1, input2)
    
    # Backward propagation to compute the gradients with respective to model output from previos step.
    out1.backward(retain_graph=True)

  def relevance_score(model, batch_size, start_layer):
    '''
    This method calculates the relevence score considered as the salience of the inputs using method proposed by Hile et al, 2021.
    '''

    # Coverting attention weight hook dictionary keys into a list
    keys = list(model.attention_weights.keys())

    if start_layer == -1:
      start_layer = len(keys) - 1

    num_tokens = model.attention_weights[keys[0]].shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=model.attention_weights[keys[0]].dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i,key in enumerate(keys):

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

    return R

  forward_backward(model, input1, input2)

  batch_size = input1.shape[0]

  input1_relevance = relevance_score(model.visual, batch_size, start_layer)[:, 0, 1:]
  CLS_idx = input2.argmax(dim=-1)
  input2_relevance = relevance_score(model.transformer, batch_size, start_layer)[:,CLS_idx, 1:CLS_idx]


  return input1_relevance, input2_relevance
