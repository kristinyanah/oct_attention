import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn

import torch.nn.functional as F


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight



class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=1, device="cuda"):
        target = target.type(torch.LongTensor).to(device)
        input_soft = F.softmax(input,dim=1)
        y2 = torch.mean(self.dice_loss(input_soft, target))
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = y1 + y2
        return y


class CombinedLoss_adv(nn.Module):
    def __init__(self):
        super(CombinedLoss_adv, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()
        
        
    #def dice_loss_new(self, predict, target): #[9, 128, 128])
        #numerator = 2 * (predict * target).sum(-1)
       # denominator = predict.sum(-1) + target.sum(-1)
      #  loss_dice = 1 - (numerator + 1) / (denominator + 1)
      #  return loss_dice.mean()
    
    def forward(self, input, target, dice_c, weight=1, device="cuda"):
        target = target.type(torch.LongTensor).to(device)
       # gt_mask  = gt_mask.type(torch.LongTensor).to(device)
        input_soft = F.softmax(input,dim=1)
        y2 = torch.mean(self.dice_loss(input_soft, target)) #0.855
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight)) #2.21
        y_fcc = y1 + y2
        z2 = dice_c #1.913
        #z1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input2, gt_mask), weight))
        y = y_fcc + z2 #4.97
        return y



class CombinedLoss_adv2(nn.Module):
    def __init__(self):
        super(CombinedLoss_adv2, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()
        
    def _get_dice(self, predict, target):    
        smooth = 1e-5    
      
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1) 
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        return score.mean()        
    
    #def forward(self, input, target, input2, gt_mask, gt_entropy, weight=1, device="cuda"):
    def forward(self, input, input2, target, weight=1, device="cuda"):
        target = target.type(torch.LongTensor).to(device)
        #gt_mask  = gt_mask.type(torch.LongTensor).to(device)
        input_soft = F.softmax(input,dim=1)
        y2 = torch.mean(self.dice_loss(input_soft, target)) #0.8531
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight)) #2.2100 2, 9, 224, 224])   [2, 224, 224]) 
       
        
       # z2 = dice_loss_new(input2, gt_mask)#1.913 torch size: [15000]
        #z3 = self._get_dice(input2, gt_mask) #0.2023
        input_soft2 = F.softmax(input2,dim=1)
        z3 = torch.mean(self.dice_loss(input_soft2, target)) #0.8570
       # z3 = self.dice_loss(input2, target)
        #gt_mask = torch.transpose(gt_mask, 1, 2)
        z1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input2, target), weight)) #4.542 [2, 9, 128, 128]) ([2, 9, 128, 128])
        y = y1 + y2 + z3 + z1 * 0.01
        return y
 #loss = criterion(target, labels.view(len(labels), 1))
