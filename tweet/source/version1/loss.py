import torch
import torch.nn as nn

class SoftArgmax(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        return None

    def forward(self, x):
        smax = self.softmax(x)
        indices = torch.arange(start=0, end=0 + x.size()[1], step=1)
        indices = indices.float()
        return torch.matmul(smax, indices)

class MSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.smax = SoftArgmax()
        return None
    
    def forward(self, start_idx, end_idx, start_logit, end_logit):
        start_logit = self.smax(start_logit).squeeze()
        end_logit = self.smax(end_logit).squeeze()
        start_loss = self.loss(start_logit, start_idx.squeeze())
        end_loss = self.loss(end_logit, end_idx.squeeze())
        total_loss = (start_loss + end_loss)

class CELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        return None
    
    def forward(self, start_logit, end_logit, start_idx, end_idx):
        start_loss = self.loss(start_logit, start_idx)
        end_loss = self.loss(end_logit, end_idx)
        total_loss = (start_loss + end_loss)
        return total_loss

class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        return None
        
    def forward(self, preds, target):
        smooth = 1.
        preds = torch.sigmoid(preds)
        iflat = preds.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )  
    
class WeightedLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = CELoss()
        return None    
    
    def forward(self, s_log, e_log, s_idx, e_idx, preds, label):
        ce_loss = self.ce_loss(s_log, e_log, s_idx, e_idx)
        dice_loss = self.dice_loss(preds, label)
        return ce_loss  # dice_loss



