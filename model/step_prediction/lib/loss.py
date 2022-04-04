import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class WHContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(WHContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s, neg_s):
        # compute image-sentence score matrix
        part_scores = get_sim(im, s)
        neg_feature = torch.stack(neg_s, 0)
        neg_scores = (im.unsqueeze(1) * neg_feature).sum(-1)

        scores = torch.cat([part_scores, neg_scores], dim=1)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(part_scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + part_scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s[:part_scores.shape[0], :part_scores.shape[1]] = cost_s[:part_scores.shape[0], :part_scores.shape[1]].masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class ContrastiveLoss3D(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss3D, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        im = im.unsqueeze(1).unsqueeze(1)
        s = s.view(1, im.shape[0], im.shape[0], -1)
        scores = (im * s).sum(-1)
        # scores = get_sim(im, s)
        diagonal = torch.stack([scores[_,_,_] for _ in range(scores.shape[0])]).view(im.size(0), 1, 1)
        # diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = torch.transpose(diagonal, 0, 1).expand_as(scores)
        d3 = torch.transpose(diagonal, 0, 2).expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_1 = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_2 = (self.margin + scores - d2).clamp(min=0)
        # image retrieval
        cost_3 = (self.margin + scores - d3).clamp(min=0)

        # clear diagonals
        mask = torch.zeros_like(scores)
        for index in range(scores.shape[0]):
            mask[index, index, index] = 1
        mask = mask > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_1 = cost_1.masked_fill_(I, 0)
        cost_2 = cost_2.masked_fill_(I, 0)
        cost_3 = cost_3.masked_fill_(I, 0)
        # cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_1 = cost_1.max(-1)[0].max(-1)[0]
            cost_2 = cost_2.max(0)[0].max(-1)[0]
            cost_3 = cost_3.max(0)[0].max(0)[0]

        # return cost_1.sum() + cost_2.sum() + cost_3.sum()
        return cost_2.sum()

class ContrastiveLoss3DTo2D(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss3DTo2D, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        im = im.unsqueeze(1).unsqueeze(1)
        s = s.view(1, im.shape[0], im.shape[0], -1)
        scores = (im * s).sum(-1)

        selected_scores = [scores[_, :, _].view(-1, 1) for _ in range(scores.shape[0])]
        scores = torch.cat(selected_scores, dim=1)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
