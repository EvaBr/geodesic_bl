#!/usr/env/bin python3.9

from typing import List, cast

import torch
import numpy as np
from torch import Tensor, einsum

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

class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        idc = kwargs["idc"] #now these are weights that we apply. 
        self.idc: List[int] = [i for i,v in enumerate(idc) if v>0]
        #If a class should be ignored, simply set weight=0 for that class.
        device = kwargs["device"]
        self.weights = torch.tensor([i for i in idc if i>0]).float().to(device)


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
