import torch
from torch.nn import functional as F

def sare_loss(outputs, Batch, N_feat, loss_type="sare_ind"):
    outputs = outputs.view(Batch, N_feat, -1)
    L = outputs.size(-1)

    output_negatives = outputs[:, 2:]
    output_anchors = outputs[:, 0]
    output_positives = outputs[:, 1]

    if (loss_type=='sare_joint'):
        dist_pos = ((output_anchors - output_positives)**2).sum(1)
        dist_pos = dist_pos.view(Batch, 1)

        output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_negatives = output_negatives.contiguous().view(-1, L)
        dist_neg = ((output_anchors - output_negatives)**2).sum(1)
        dist_neg = dist_neg.view(Batch, -1)

        dist = - torch.cat((dist_pos, dist_neg), 1)
        dist = F.log_softmax(dist, 1)
        loss = (- dist[:, 0]).sum()

    elif (loss_type=='sare_ind'):
        dist_pos = ((output_anchors - output_positives)**2).sum(1)
        dist_pos = dist_pos.view(Batch, 1)

        output_anchors = output_anchors.unsqueeze(1).expand_as(output_negatives).contiguous().view(-1, L)
        output_negatives = output_negatives.contiguous().view(-1, L)
        dist_neg = ((output_anchors - output_negatives)**2).sum(1)
        dist_neg = dist_neg.view(Batch, -1)

        dist_neg = dist_neg.unsqueeze(2)
        dist_pos = dist_pos.view(Batch, 1, 1).expand_as(dist_neg)
        dist = - torch.cat((dist_pos, dist_neg), 2).view(-1, 2)
        dist = F.log_softmax(dist, 1)
        loss = (- dist[:, 0]).sum()
    else:
        print("WARNING: WRONG LOSS TYPE")
        loss = 0
    return loss