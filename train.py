import sys
import torch
import torch.nn as nn
from loss import MultiMarginLoss, LDAMLoss, CB_loss, CenterLoss, kw_rank_loss
from model import Vivit_backbone, ResnetTCN
import copy
from data import VivitProcessorToTensor
from transformers import VivitImageProcessor
import numpy as np

class MultiEngagementPredictor(nn.Module):
    def __init__(self, model_name, loss_type, second_loss_type = None, CUDA=0, labels = 0):
        super().__init__()
        self.loss_type = loss_type
        self.second_loss_type = second_loss_type
        self.CUDA = CUDA
        self.model_name = model_name
        self.m = 0.999
        output_dim = 1 if loss_type in ["mocorank", "mse"] else 4
        cls_num_list = [labels.count(0), labels.count(1), labels.count(2), labels.count(3)]
    
        #Define Model
        self.model_k = None
        if model_name == "Vivit":
            self.model_q= Vivit_backbone(output_dim=output_dim)
            self.model_k= Vivit_backbone(output_dim=output_dim)
            self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
            self.processor = VivitProcessorToTensor()
        if model_name == "ResnetTCN":
            self.model_q = ResnetTCN(128,128,output_dim)
            self.model_k= ResnetTCN(128,128,output_dim)

        #Moco
        if loss_type == "mocorank":
            self.model_k.eval()

            for param_q, param_k in zip(
                self.model_q.parameters(), self.model_k.parameters()
            ):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False

            #prepare the pool (moco)
            queue_size = 512*4 
            rand_index = torch.randperm(queue_size)
            self.register_buffer("queue", torch.cat((torch.ones(1,queue_size//4)*(-0.75), torch.ones(1,queue_size//4)*(-0.25), torch.ones(1,queue_size//4)*(0.25), torch.ones(1,queue_size//4)*0.75),1))
            init_label = torch.tensor([0 for _ in range(queue_size//4)] + [1 for _ in range(queue_size//4)] + [2 for _ in range(queue_size//4)] + [3 for _ in range(queue_size//4)])
            self.register_buffer("queue_label", init_label[rand_index])
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_lstm", torch.zeros((queue_size,128), dtype=torch.float)) #*
            self.queue[0] = self.queue[0][rand_index]

            #Loss
            self.criterion_multimargin = MultiMarginLoss(tol=0.5, flexible_margin=True, CUDA=self.CUDA)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "cbloss_focal":
            self.criterion = CB_loss(cls_num_list, "focal", CUDA)
        elif loss_type == "cbloss_ce":
            self.criterion = CB_loss(cls_num_list, "softmax", CUDA)
        elif loss_type == "ldam":
            idx=0
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(CUDA)
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=None, cuda = CUDA).to(CUDA)
        
        if second_loss_type == "rankloss":
            self.second_criterion = kw_rank_loss()
        elif second_loss_type == "triplet":
            self.second_criterion = nn.TripletMarginWithDistanceLoss(margin=1)
        elif second_loss_type == "centerloss":
            self.second_criterion = CenterLoss(4, 128, CUDA)

    def forward(self,x):
        pred_q, feat_q = self.model_q(x)
        with torch.no_grad():
            if self.loss_type == "mocorank":
                pred_k, feat_k = self.model_k(x)
            else:
                pred_k, feat_k = None, None

        return pred_q, feat_q, pred_k, feat_k
  
    def training_step(self, x, label, video):
        anchor_label = label.to(self.CUDA)

        if self.model_name == "Vivit":
            v_tensor = [self.processor(v) for v in video]
            inputs = self.image_processor(v_tensor, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(self.CUDA)
        else:
            x = x.to(self.CUDA)

        #update key model (normal moco)
        if self.loss_type == "mocorank":
            self._momentum_update_key_encoder()
        
        #forward aus prediction
        if self.model_name == "Vivit":
            pred_q, feat_q, pred_k, feat_k = self.forward(inputs)
        else:
            pred_q, feat_q, pred_k, feat_k = self.forward(x)

        #Loss
        if self.loss_type == "mocorank":
            loss = self.criterion_multimargin(self.queue, self.queue_label, self.queue_lstm, pred_q, anchor_label, feat_q) #*
            self._dequeue_and_enqueue_queue(pred_k,feat_k,anchor_label) #*
        elif self.loss_type == "mse":
            scaled_label = (anchor_label/2)-0.75 #[-0.75, -0.25, 0.25, 0.75]
            loss = self.criterion(pred_q, scaled_label.unsqueeze(dim=1))
        elif self.loss_type in ["ldam", "cbloss_focal", "cbloss_ce", "ce"]:
            loss = self.criterion(pred_q, anchor_label)
        else:
            print("loss %s not implemented" % self.loss_type)
            sys.exit()

    
        #Additional loss
        if self.second_loss_type == "rankloss":
            loss_add = self.second_criterion(feat_q, anchor_label)
        elif self.second_loss_type == "triplet":
            pos_x, neg_x = None, None
            _,feat_pos,_,_ = self.forward(pos_x)
            _,feat_neg,_,_= self.forward(neg_x)
            loss_add = self.second_criterion(feat_q, feat_pos, feat_neg)
        elif self.second_loss_type == "centerloss":
            loss_add = self.second_criterion(feat_q, anchor_label)
        else:
            loss_add = 0      
        
        return loss, loss_add
  
    def validation_step(self, x, label, video):
        if self.model_name == "Vivit":
            v_tensor = [self.processor(v) for v in video]
            inputs = self.image_processor(v_tensor, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(self.CUDA)
        else:
            x = x.to(self.CUDA)

        #forward
        if self.model_name == "Vivit":
            pred_q, feat_q, pred_k, feat_k = self.forward(inputs)
        else:
            pred_q, feat_q, pred_k, feat_k = self.forward(x)
        
        if self.loss_type in ["ldam", "cbloss_focal", "cbloss_ce", "ce"]:
            score = torch.softmax(pred_q, dim=1).argmax(dim=1)
        else:
            score = pred_q[:,0]
            
        return label ,score

    def _dequeue_and_enqueue_queue(self, keys, keys_lstm, keys_label):  #*
        '''
        Update the queue (moco)
        '''
        batch_size = keys.shape[0]
        key_length = self.queue.shape[1]

        ptr = int(self.queue_ptr)
        if key_length % batch_size != 0:
            return
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_lstm[ptr : ptr+batch_size, :] = keys_lstm #*
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.queue_label[ptr : ptr+batch_size] = keys_label
        ptr = (ptr + batch_size) % key_length  # move pointer

        self.queue_ptr[0] = ptr
    
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder (moco)
        """
        for param_q, param_k in zip(
            self.model_q.parameters(), self.model_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
