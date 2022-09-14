import torch


def salience(input1, input2, model, device):

  def forward_backward(model, input1, input2):
    out1, out2 = model(input1, input2)
    out1.backward(retain_graph=True)

  def relevance_score(model, batch_size, reverse=True):
    keys = list(model.attention_weights.keys())

    if reverse:
      keys = keys[-1::]

    num_tokens = model.attention_weights[keys[0]].shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=model.attention_weights[keys[0]].dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i,key in enumerate(keys):
      grad = model.attention_grads[key].detach()
      cam = model.attention_weights[key].detach()
      cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
      grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
      cam = grad * cam
      cam = cam.reshape(batch_size, -1, cam.shape[-1]. cam.shape[-1])
      cam = cam.clamp(min=0).mean(dim=1)
      R = R + torch.bmm(cam, R)

    return R

  forward_backward(model, input1, input2)

  batch_size = input1.shape[0]

  input1_relevance = relevance_score(model.visual, batch_size, True)[:, 0, 1:]
  input2_relevance = relevance_score(model.transformer, batch_size)

  return input1_relevance, input2_relevance
