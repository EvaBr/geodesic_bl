#!/usr/bin/env python3.9

from typing import Any, Callable, Tuple, List
from operator import add

from utils import map_, uc_


class DummyScheduler(object):
    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        return optimizer, loss_fns, loss_weights


class AddWeightLoss():
    def __init__(self, to_add: List[float]):
        self.to_add: List[float] = to_add

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        assert len(self.to_add) == len(loss_weights[0])
        if len(loss_weights) > 1:
            raise NotImplementedError
        new_weights: List[List[float]] = map_(lambda w: map_(uc_(add), zip(w, self.to_add)), loss_weights)

        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights


class StealWeight():
    def __init__(self, to_steal: float):
        self.to_steal: float = to_steal

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
    #    new_weights: List[List[float]] = [[max(0.1, a - self.to_steal), b + self.to_steal] for a, b in loss_weights]

        #new_weights: List[List[float]] = [[max(0.01, (1 - self.to_steal/one_list[0])*a) for a in one_list[:-1]] +[min(one_list[-1] + self.to_steal, 1.0)] for one_list in loss_weights]
        new_weights: List[List[float]] = [[max(0.01, a - self.to_steal) for a in one_list[:-1]] +[min(one_list[-1] + self.to_steal, 1.0)] for one_list in loss_weights]


        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights


class BLbasedWeight():
    def __init__(self, to_add: List[float], **kwargs): # bl_thr: float
        self.upper_lim: float = kwargs['upper_lim'] if 'upper_lim' in kwargs else [1000 for i in to_add] #whatevs, just to avoid reaching it.
        self.to_add: List[float] = to_add
        self.stop: bool = False #whether to put Dice / other non-BL loss contributions to 0

        #self.bl_thr: List[float] = bl_thr <- should implement this version if we wish to do early stopping depending on individual BL losses/contributions
        #self.bl_thr: float = bl_thr <- better do compare withthreshold inside main, so we dont need access to current los values in here

    def __call__(self, epoch: int, optimizer: Any, loss_fns: List[List[Callable]], loss_weights: List[List[float]]) \
            -> Tuple[float, List[List[Callable]], List[List[float]]]:
        assert len(self.to_add) == len(loss_weights[0]) #should be the same also for loss_weights[i] tho, as implemented atm
        #assert len(self.bl_thr)<len(self.to_add) #<= if we assume we can train with only BL
        
        new_weights: List[List[float]] = \
            [[min(ulim, max(0, a+plus)) for a,plus,ulim in zip(one_list,self.to_add,self.upper_lim)] for one_list in loss_weights]

        if self.stop:
            for i in range(len(new_weights)):
                for j in range(len(new_weights[i])):
                    if not loss_fns[i][j].__class__.__name__=="SurfaceLoss": #TODO: instead of retaining all surface losses, have a variable to controll which losses ghet removed
                        new_weights[i][j]=0



        print(f"Loss weights went from {loss_weights} to {new_weights}")

        return optimizer, loss_fns, new_weights
