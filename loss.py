import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    def __init__(self, num_class=4, num_feature=128,cuda=None):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.num_feature = num_feature
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).to(cuda))

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)
        return loss

class kw_rank_loss(nn.Module):    
    def __init__(self):        
        super(kw_rank_loss, self).__init__()          
    def forward(self, feature_results, target_var):
        engagement_level_1 = np.zeros((1, feature_results.shape[1]))
        engagement_level_2 = np.zeros((1, feature_results.shape[1]))
        engagement_level_3 = np.zeros((1, feature_results.shape[1]))
        engagement_level_4 = np.zeros((1, feature_results.shape[1]))
        level_1_num = level_2_num = level_3_num = level_4_num = 0
        #pdb.set_trace()
        size = feature_results.shape[0]
        feature_results = feature_results.cpu().data.numpy()
        margin = 0.75
        #pdb.set_trace()
        for i in range(size):
            if target_var[i] <= 0.1:
                engagement_level_1 += feature_results[i]
                level_1_num += 1
            if 0.3 <target_var[i] <= 0.4:
                engagement_level_2 += feature_results[i]
                level_2_num += 1
            if 0.6 <target_var[i] <= 0.7:
                engagement_level_3 += feature_results[i]
                level_3_num += 1
            if 0.8 <target_var[i] <= 1.1:
                engagement_level_4 += feature_results[i]
                level_4_num += 1
        #pdb.set_trace()
        engagement_level_1 = engagement_level_1/level_1_num if level_1_num != 0 else engagement_level_1
        engagement_level_2 = engagement_level_2/level_2_num if level_2_num != 0 else engagement_level_2
        engagement_level_3 = engagement_level_3/level_3_num if level_3_num != 0 else engagement_level_3
        engagement_level_4 = engagement_level_4/level_4_num if level_4_num != 0 else engagement_level_4
        #pdb.set_trace()
        dist_1_2 = np.linalg.norm(engagement_level_1 - engagement_level_2)
        dist_1_3 = np.linalg.norm(engagement_level_1 - engagement_level_3)
        dist_1_4 = np.linalg.norm(engagement_level_1 - engagement_level_4)
        dist_2_3 = np.linalg.norm(engagement_level_2 - engagement_level_3)
        dist_2_4 = np.linalg.norm(engagement_level_2 - engagement_level_4)
        dist_3_4 = np.linalg.norm(engagement_level_3 - engagement_level_4)
        #pdb.set_trace()
        loss = max(0.0,(dist_1_2 - dist_1_3 + margin)) + max(0.0,(dist_1_2 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_2_3 - dist_2_4 + margin)) + max(0.0,(dist_2_3 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_3_4 - dist_2_4 + margin)) + max(0.0,(dist_3_4 - dist_1_4 +2*margin))       
        return loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, cuda=0):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list.to(cuda)
        assert s > 0
        self.s = s

        if weight == None:
            self.weight = None
        else:
            self.weight = weight.to(cuda)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, loss_type="focal",CUDA=0, no_of_classes=4, beta=0.9999, gamma=2):
        super(CB_loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.gamma = gamma
        self.cuda = CUDA
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        self.weights = weights / np.sum(weights) * no_of_classes

    def forward(self, logits, labels):
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()
        weights = torch.tensor(self.weights).float().to(self.cuda)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss

class MultiMarginLoss(nn.Module):
    def __init__(self, tol=0, flexible_margin=False, CUDA=0):
        super(MultiMarginLoss, self).__init__()
        loss_L1 = nn.L1Loss()
        loss_margin1 = nn.MarginRankingLoss(margin=0.5-tol)
        loss_margin2 = nn.MarginRankingLoss(margin=1.0-tol)
        loss_margin3 = nn.MarginRankingLoss(margin=1.5-tol)
        self.loss_margin = [loss_L1, loss_margin1, loss_margin2, loss_margin3]
        self.flexible_margin = flexible_margin
        self.tol = tol
        self.CUDA = CUDA

    def forward(self, queue, queue_label, queue_lstm, anchor_mlp, anchor_label, anchor_lstm):  #*
        #queue: (1,512)
        #queue_label: (512)
        #anchor_mlp: (64,1)
        #anchor_label: (64)
        #queue_lstm: (512,128)
        #anchor_lstm: (64,128)

        label_diff = anchor_label.unsqueeze(1)-queue_label #(64,512)
        score_diff = anchor_mlp - queue
        margin_mask = torch.abs(label_diff)  
        target_mask = (label_diff).sign()
        

        if self.flexible_margin:
            with torch.no_grad():
                cos_sim = F.cosine_similarity(anchor_lstm.unsqueeze(1), queue_lstm, dim=-1) #(64,512)
                cos_sim = (cos_sim+1)/2
            
        loss = 0
        for gap in range(0,4):
            scores1 = score_diff[margin_mask==gap]
            target = target_mask[margin_mask==gap]
            
            if len(target) == 0:
                continue

            if gap != 0:
                if self.flexible_margin:
                    scores2 = (cos_sim[margin_mask==gap]*self.tol * target)
                else:
                    scores2 = torch.zeros(scores1.shape).to(self.CUDA)
                loss += self.loss_margin[gap](scores1, scores2, target)
                #equivalent to max(0, (s2-model(x)) + (0,0.5,1.0) + 0.5(cos+1)/2) when l1 > l2
                #equivalent to max(0, (model(x)-s2) + (0,0.5,1.0)+ 0.5(cos+1/2)) when l2 < l1
            else:
                scores2 = torch.zeros(scores1.shape).to(self.CUDA)
                loss += self.loss_margin[gap](scores1, scores2)

        return loss