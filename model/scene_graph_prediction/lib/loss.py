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

    def forward(self, im, s, goal_ids=None):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        if goal_ids is not None:
            goal_masks = scores.new_ones((im.shape[0], s.shape[0]))
            device = im.get_device()
            dtype = im.dtype
            start_index = 0
            for index in range(len(goal_ids)):
                if goal_ids[start_index] != goal_ids[index]:
                    end_index = index
                    partial_mask = torch.eye(end_index - start_index, dtype=dtype, device=device)
                    goal_masks[start_index:end_index, start_index:end_index] = partial_mask
                    start_index = end_index
            end_index = len(goal_ids)
            partial_mask = torch.eye(end_index - start_index, dtype=dtype, device=device)
            goal_masks[start_index:end_index, start_index:end_index] = partial_mask
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        if goal_ids is not None:
            cost_s *= goal_masks
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        if goal_ids is not None:
            cost_im *= goal_masks

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

