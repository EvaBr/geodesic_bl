#!/usr/env/bin python3.9

from typing import List, cast

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from torch.autograd import Function
from torch.autograd import Variable

from bilateralfilter import bilateralfilter_batch

from utils import simplex, probs2one_hot, one_hot
from utils import one_hot2hd_dist


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class JointBLandCE():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        log_notp: Tensor = (1. - probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        dc: Tensor = dist_maps[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10 #this is the CE part

        

        multipled = einsum("bkwh,bkwh->bkwh", probs, dc)
        loss += multipled.mean() #this is the BL part


        loss2 = - einsum("bkwh,bkwh->bkwh", 1.-mask, log_notp)
        loss2 = einsum("bkwh,bkwh->", loss2, dc)/(mask.numel()-mask.sum() + 1e-10)

        return loss + loss2
         

class NoClassCrossEntropy(): #to be used on conjunction with BL and CE
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs) 
        assert not one_hot(dist_maps)

        log_notp: Tensor = (1. - probs[:, self.idc, ...] + 1e-10).log()
        dc: Tensor = cast(Tensor, dist_maps[:, self.idc, ...].type(torch.float32))
        mask: Tensor = torch.count_nonzero(torch.greater(dc, 0)).float() #, dim = 1)
        #here dc would actually need to be dc[idc] X (1-gt[idc]), that's why we calc mask 


        loss = - einsum("bkwh,bkwh->", dc, log_notp)
        loss /= (dc.numel()-mask + 1e-10) #could use just numel here, since mask will be small when training with points

        return loss
        


class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        self.idc = kwargs["idc"] 
        weights = kwargs["weights"] #now these are weights that we apply. Should be one for each idc!
        assert len(self.idc)==len(weights), print("Weights should match the provided indices!")
        device = kwargs["device"] #does not need to be specified manually, is given internally during loss setup
        self.weights = torch.tensor(weights).float().to(device)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->c", mask, log_p)
        loss = torch.dot(loss, self.weights)

        mask = einsum("bcwh->c", mask)
        mask = torch.dot(mask, self.weights)
        #loss /= max(mask.sum(), 1e-10) #mask.sum() + 1e-10
        loss /= mask + 1e-10
        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class WeightedGeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        idc = kwargs["idc"] #now these are weights that we apply. 
        self.idc: List[int] = [i for i,v in enumerate(idc) if v>0]
        self.opt: int = kwargs["opt"] if "opt" in kwargs else 2
        #If a class should be ignored, simply set weight=0 for that class.
        device = kwargs["device"]
        self.weights = torch.tensor([i for i in idc if i>0]).float().view(1, len(self.idc)).to(device)

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        #OPTION 1: instead of dynamically changing weights batch-based, keep them static based on input weights
        if self.opt==1:
            w: Tensor = 1 / ((self.weights+1e-10)**2)
            intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
            union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

            divided: Tensor = 1 -  (2*einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        #OPTION 2: imitate the computation that happens if you put in multiple/per-class GDL losses as args 
        else: #if self.opt==2:
            w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
            intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
            union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

            divided: Tensor = self.weights.sum() - 2 * einsum("bk->b", (intersection + 1e-10) / (union + 1e-10) * self.weights)

        loss = divided.mean()

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


class WeightedSurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        idc = kwargs["idc"] #now these are weights that we apply. 
        #If a class should be ignored, simply set weight=0 for that class.
        self.idc: List[int] = [i for i,v in enumerate(idc) if v>0]

        n_classes = kwargs["n_classes"] if "n_classes" in kwargs else 7 #assume mostly use for POEM data
        device = kwargs["device"]
        self.weights = torch.tensor([i for i in idc if i>0]).float().to(device)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        
        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        #OPTION 1: do a soooort-of weighted mean by hand?
    #    weightedall = torch.dot(einsum("bkwh->k", pc), self.weights)
    #    weighted = torch.dot(einsum("bkwh->k", multipled), self.weights)
    #    loss = weighted / (weightedall + 1e-10) #kind of weighted mean? 

        #OPTION 2: Simulate  the computation that happens if you put in multiple/per-class BLs in args
        loss = torch.dot(multipled.mean(dim=(0,2,3)), self.weights) 
        
        return loss


BoundaryLoss = SurfaceLoss
WeightedBoundaryLoss = WeightedSurfaceLoss


class HausdorffLoss():
    """
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs)
        assert simplex(target)
        assert probs.shape == target.shape

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))
        assert pc.shape == tc.shape == (B, len(self.idc), *xyz)

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)
        assert target_dm_npy.shape == tc.shape == pc.shape
        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)
        assert pred_dm_npy.shape == tc.shape == pc.shape
        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc)**2
        dtm = tdm**2 + pdm**2

        multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)

        loss = multipled.mean()

        return loss


class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bkwh,bkwh,bkwh->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class DenseCRFLossFunction(Function):
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape

        ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None


class DenseCRFLoss(nn.Module):
    def __init__(self, **kwargs):
        super(DenseCRFLoss, self).__init__()
        self.weight = kwargs["weight"]
        self.sigma_rgb = kwargs["sigma_rgb"]
        self.sigma_xy = kwargs["sigma_xy"]
        self.scale_factor = kwargs["scale_factor"]

    def forward(self, segmentations, images):
        """ scale imag by scale_factor """
        B, C, W, H = images.shape
        assert C == 3, images.shape
        B_, K, W_, H_ = segmentations.shape
        assert B == B_, (B, B_)
        assert W == W_, (W, W_)
        assert H == H_, (H, H_)

        # ROIs = torch.ones_like(images)
        ROIs = torch.ones((B, W, H))

        scaled_images = F.interpolate(images.cpu(), scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1), scale_factor=self.scale_factor).squeeze(1)

        return self.weight * DenseCRFLossFunction.apply(scaled_images,
                                                        scaled_segs,
                                                        self.sigma_rgb,
                                                        self.sigma_xy * self.scale_factor,
                                                        scaled_ROIs)[0]

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
