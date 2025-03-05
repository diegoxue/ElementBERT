# TODO debug & description

from copy import deepcopy
import itertools
import math
from typing import Dict, List, Tuple
import uuid
import sys

import joblib
import torch
import numpy as np
from loguru import logger
from scipy.optimize import minimize, LinearConstraint

from environment import (COMPOSITION_ROUNDUP_DIGITS, COMPOSITION_INTERVAL,
                         TI_MIN, TI_MAX, NI_MIN, NI_MAX, CU_MIN, CU_MAX,
                         HF_MIN, HF_MAX)
from composition_bo import AbstractCompositionBayesianOptimization, ensure_rng

class ListSet():
    def __init__(self):
        self._data_list = []
        self._data_set = set()
    
    def encode(self, comp: List[float]) -> str:
        return '*'.join(map(str, comp))
    
    def extend(self, comps: List[List[float]]):
        for c in comps:
            c = np.array(c).round(COMPOSITION_ROUNDUP_DIGITS)
            c_key = self.encode(c)
            if c_key not in self._data_set:
                self._data_list.append(c)
                self._data_set.add(c_key)
    
    def get(self) -> List[List[float]]:
        return self._data_list

class GradOptBayesianOptimization(AbstractCompositionBayesianOptimization):
    _comp_tolerance = 1e-6
    _default_inner_opt_num = 80
    _batch_q = 1

    ''' for debug purposes '''
    def feature_selection(self):
        super().feature_selection()

        self._sel_elem_features = torch.from_numpy(self._sel_elem_features).requires_grad_(False)

    def cal_elemental_feature(self, comp) -> torch.Tensor:
        ''' Pytorch tensorized version for calculation of weighted average elemental features '''
        if not isinstance(comp, torch.Tensor):
            comp = torch.from_numpy(np.array(comp)).requires_grad_(False)
        
        return torch.matmul(comp, self._sel_elem_features)
    
    def update_elem_feat_bounds(self, elem_feat_arr: torch.Tensor):
        ''' Pytorch tensorized version for updating bounds of elemental features '''
        _feat_min_bounds = elem_feat_arr.min(dim = 0).values
        _feat_max_bounds = elem_feat_arr.max(dim = 0).values
        
        _zero_mask = (_feat_min_bounds == _feat_max_bounds)
        _feat_max_bounds[_zero_mask] = _feat_max_bounds[_zero_mask] + 1

        self._elem_feat_bounds = torch.vstack((_feat_min_bounds, _feat_max_bounds)).detach().requires_grad_(False)

    ''' Re-use parent's .standardize func. Same broadcast logic for pytorch & numpy. '''
    # def standardize(...): pass


    def ucb_acqf(self, mean: torch.Tensor, std: torch.Tensor, kappa = 1.92) -> torch.Tensor:
        return mean + kappa * std
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' 
            GPR prediction.
            Parameters
            ----------
            elem_feat: numpy array
                unseen unnormalized elemental feature 
        '''
        x = self.standardize(x.reshape(-1, self._sel_elem_feat_num))
        y_posterior = self._gpr.posterior(x)
        y_mean = y_posterior.mean
        y_std = y_posterior.variance.sqrt()
        return y_mean.flatten(), y_std.flatten()

    def generate_random_initial_guess(self):
        ''' Generate one random composition '''
        random_state = ensure_rng(None)

        while True:
            _comp = np.zeros((len(self._bounds), ))
            # Ti
            _comp[0] = random_state.uniform(1 - NI_MAX, TI_MAX)
            # Ni
            _comp[1] = 1.0 - _comp[0]

            _flag = True    # found flag
            for _elem_idx in range(2, self._elem_num):
                _sub_idx = 0 if _elem_idx in [3, 5] else 1  # if Hf or Zr, sub Ti
                _comp[_elem_idx] = random_state.uniform(*(self._bounds[_elem_idx]))
                _comp[_sub_idx] -= _comp[_elem_idx]

                _comp = self.round(_comp)
                if _comp[_elem_idx] < self._bounds[_elem_idx][0] or \
                        _comp[_elem_idx] > self._bounds[_elem_idx][1] or \
                        _comp[_sub_idx] < self._bounds[_sub_idx][0] or \
                        _comp[_sub_idx] > self._bounds[_sub_idx][1]:
                    _flag = False
                    break
            
            if _flag:
                return _comp
            else:
                continue

    def round_composition(self, comp: List[float]) -> List[float]:
        ''' Round an arbitrary composition to the pre-defined composition accuracy. '''
        def _single_comp_round(_c, _elem_idx):
            x_low_neighbor = math.floor(_c / COMPOSITION_INTERVAL) * COMPOSITION_INTERVAL
            x_high_neighbor = math.ceil(_c / COMPOSITION_INTERVAL) * COMPOSITION_INTERVAL
            return np.unique(np.clip([x_low_neighbor, x_high_neighbor], *(self._bounds[_elem_idx])))
        
        _usable_dis_f = [_single_comp_round(_c, _elem_idx) for _elem_idx, _c in enumerate(comp)]
        _potential_comps = np.array(list(itertools.product(*_usable_dis_f)))
        mask = abs(_potential_comps.sum(axis = -1) - 1.0) < self._comp_tolerance

        return _potential_comps[mask]

    def f(self, comp):
        ''' composition -> atomic features -> standardized -> mean, std -> (acq_func_val, acq_func_grad_val) '''
        comp_tensor = (
            torch.from_numpy(comp)
            .contiguous()
            .requires_grad_(True)
        )
        a_feat = self.cal_elemental_feature(comp_tensor).view(1, -1)
        mean, std = self.predict(a_feat)
        neg_acqf_val = - self.ei_acqf(mean, std)
        # compute gradient of acq_func w.r.t. composition
        gradf = torch.autograd.grad(neg_acqf_val, comp_tensor)[0].contiguous().view(-1).numpy() # TODO: check
        fval = neg_acqf_val.item()

        # acq_func_val, acq_func_grad_val
        return fval, gradf

    def suggest_next_x(self) -> List[float]:
        ''' Inner argmax of acquisition function by enumeration of all possible compositions '''
        
        ''' scipy constraints for alloy compositions '''
        constraints = [
            {   # sum = 1.0
                'type':'eq', 
                'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 1.0, 
                'jac': lambda x: np.ascontiguousarray(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            },
            {   # sum(Ni, ...) <= NI_MAX
                'type':'ineq', 
                'fun': lambda x: NI_MAX - x[1] - x[2] - x[4], 
                'jac': lambda x: np.ascontiguousarray(np.array([0.0, -1.0, -1.0, 0.0, -1.0, 0.0]))
            },
            {   # sum(Ti, ...) <= TI_MAX
                'type':'ineq', 
                'fun': lambda x: TI_MAX - x[0] - x[3] - x[5], 
                'jac': lambda x: np.ascontiguousarray(np.array([-1.0, 0.0, 0.0, -1.0, 0.0, -1.0]))
            },
        ]

        _comp_bounds_tensor = torch.tensor(self._bounds, dtype = torch.double).requires_grad_(False)

        _rounded_comps = ListSet()
        for opt_no in range(self._default_inner_opt_num):
            ''' single optimization '''
            x0 = self.generate_random_initial_guess()

            ''' scipy optimize '''
            res = minimize(
                self.f,
                x0,
                method = "SLSQP",
                jac = True,                         # implement f() -> (f_val, f_grad_val)
                bounds = _comp_bounds_tensor,       # no need for scipy.optimize.Bounds
                constraints = constraints,
            )

            _rounded_comps.extend(self.round_composition(x0))       # warm-ups
            _rounded_comps.extend(self.round_composition(res.x))    # grad-optimized

        ''' calculate EI vals for rounded comps '''
        _rounded_comps = np.array(_rounded_comps.get())
        _rounded_comps_tensor = torch.from_numpy(_rounded_comps)
        a_feat = self.cal_elemental_feature(_rounded_comps_tensor)
        mean, std = self.predict(a_feat)
        ei_vals = self.ei_acqf(mean, std).detach().numpy().flatten()
        
        # find the most promising un-tested comp
        _desc_idxs = np.argsort(ei_vals)[::-1]
        for _i in _desc_idxs:
            _comp = _rounded_comps[_i]
            if not self.contains(_comp):
                return _comp

        raise Exception('Big logic leak, should never run this line.')
    
    def run_batch(self):
        ''' Batch version '''
        _local_id = str(uuid.uuid4())[:8]

        ''' 1. identify important elemental feature subset '''
        self.feature_selection()

        ''' 2. BO outer loop '''
        while(self.exp_round) < self._total_exp_num:
            ''' 3. Imaginary (sequential) batch iter '''
            for _ in range(self._batch_q):
                _cal_feature = self.cal_elemental_feature(self._exp_comp_history)
                _target = np.array(self._exp_target_history)

                self.fit(_cal_feature, _target)
            
                # imaginary 'experiments'
                _next_comp = self.suggest_next_x()
                _next_comp_tensor = torch.tensor(_next_comp).reshape(1, -1)
                _a_feat = self.cal_elemental_feature(_next_comp_tensor)
                _next_target = self.predict(_a_feat)[0].item()

                self.register(_next_comp, _next_target)

            ''' 4. Update to real-world experimental interactions '''
            for _img_comp_idx in range(self.exp_round - self._batch_q, self.exp_round):
                _comp = self._exp_comp_history[_img_comp_idx]
                _target = self._ground_truth_func(_comp)
                self._exp_target_history[_img_comp_idx] = _target

            ''' 5. log '''
            logger.info(f'{_local_id} - round {self.exp_round}, best-so-far {self.best_so_far}')

    def save(self):
        joblib.dump(
            (self._exp_comp_history, self._exp_target_history),
            f'bt-c6-{self._init_seed}-exp-{self._init_rand_num}-{self._total_exp_num}-{str(uuid.uuid4())[:8]}.pkl'
        )

def serial_job(random_seed):
    gradopt_BO = GradOptBayesianOptimization(
        init_seed=random_seed, 
    )
    gradopt_BO.run()
    gradopt_BO.save()

if __name__ == '__main__':
    random_seed = int(sys.argv[1])

    serial_job(random_seed)