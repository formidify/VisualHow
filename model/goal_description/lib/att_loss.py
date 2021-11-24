import torch
import torch.nn.functional as F
eps = 1e-15

# attention CE
def attention_loss(pred,gt):
    batch = len(pred)
    pred = pred.view(batch,-1)
    gt = gt.view(batch,-1)
    loss = -(gt*torch.log(torch.clamp(pred,min=eps,max=1))).sum(-1)
    return loss.mean()

# attention balanced BCE
def attention_loss_bce(pred,gt):
    batch = len(pred)
    pred = pred.view(batch,-1)
    gt = gt.view(batch,-1)
    neg_weight = gt.mean(-1,keepdim=True)
    pos_weight = 1-neg_weight
    loss = -(pos_weight*gt*torch.log(torch.clamp(pred,min=eps,max=1))+neg_weight*(1-gt)*torch.log(torch.clamp(1-pred,min=eps,max=1))).sum(-1)
    return loss.mean()


# attention KLD
def attention_loss_kld(pred,gt):
    batch = len(pred)
    pred = pred.view(batch,-1)
    gt = gt.view(batch,-1)
    loss = torch.mul(gt,torch.log(torch.div(gt,pred+eps) + eps)).sum(-1)
    return loss.mean()

# attentive feature matching
def attention_loss_feat(pred,gt,feat):
    # feat = feat.detach() # may disable feature training to avoid trivial solutions
    batch = len(pred)
    pred = pred.view(batch,-1,1)
    gt = gt.view(batch,-1,1)
    seq_len = pred.size(1)
    feat = feat.view(batch,seq_len,-1)
    pred_feat = torch.mul(pred,feat).sum(1)
    gt_feat = torch.mul(gt,feat).sum(1)
    loss =  F.cosine_similarity(pred_feat, gt_feat,dim=-1)

    return (1-loss).mean()
