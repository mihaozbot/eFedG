
import torch
import torch.nn as nn
#from torch.nn import Parameter
import math


class ModelOps:
    def __init__(self, parent):
        self.parent = parent
        
        self.parent.debug_flag = 0
        self.parent.enable_debugging = False
        self.parent.enable_adding = True
        self.parent.enable_merging = True

        self.par_lim = {
                'kappa_n': [1, self.parent.feature_dim],
                'num_sigma': [1, 2 * self.parent.feature_dim],
                'kappa_join': [0.3, 10],
                'N_r': [1, 2 * self.parent.feature_dim],
                'kappa_features': [0, 1],
                'c_max': [self.parent.num_classes, 100*self.parent.num_classes],
            }
    @torch.no_grad()
    def ensure_capacity(self, new_c):
  
        # Determine whether to expand or contract the model capacity
        should_expand = new_c >= self.parent.current_capacity
        should_contract = (new_c > 2 * self.parent.c_max) and (self.parent.current_capacity > 2 * self.parent.N_r) and (new_c < (self.parent.current_capacity / 2 - 1))

        if should_expand or should_contract:
            self.modify_capacity(new_c)
    
            
    def modify_capacity(self, new_c):
        
        new_capacity = 2 ** math.ceil(math.log2(new_c))  # Find the next power of two greater than the given number
        
        self.parent.mu =(self._resize_tensor(self.parent.mu, (new_capacity, self.parent.feature_dim)))
        self.parent.mu_og = (self._resize_tensor(self.parent.mu_og, (new_capacity, self.parent.feature_dim)))
        self.parent.S = self._resize_tensor(self.parent.S, (new_capacity, self.parent.feature_dim, self.parent.feature_dim))
        self.parent.n = (self._resize_tensor(self.parent.n, (new_capacity,)))
        self.parent.S_inv = self._resize_tensor(self.parent.S_inv, (new_capacity, self.parent.feature_dim, self.parent.feature_dim))
        
        if 0:
            self.parent.P = self._resize_tensor(self.parent.P, (new_capacity, self.parent.n_phi, self.parent.n_phi))
            self.parent.theta = nn.Parameter(self._resize_tensor(self.parent.theta, (new_capacity, self.parent.n_phi, self.parent.num_classes )))
            
        self.parent.cluster_labels = self._resize_tensor(self.parent.cluster_labels, (new_capacity,self.parent.num_classes))
        self.parent.score = self._resize_tensor(self.parent.score, (new_capacity,))
        self.parent.num_pred = self._resize_tensor(self.parent.num_pred, (new_capacity,))
        self.parent.age = self._resize_tensor(self.parent.age, (new_capacity,))

        self.parent.current_capacity = new_capacity

    def _resize_tensor(self, old_tensor, new_size):
        new_tensor = torch.zeros(new_size, dtype=old_tensor.dtype, device=old_tensor.device, requires_grad=False)
        new_tensor[:self.parent.c] = old_tensor[:self.parent.c] 
        return new_tensor
    
    def toggle_adding(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_adding = not self.parent.enable_adding
            state = "enabled" if self.parent.enable_adding else "disabled"
        else:
            self.parent.enable_adding = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Cluster adding has been {state}.")

    def toggle_merging(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_merging = not self.parent.enable_merging
            state = "enabled" if self.parent.enable_merging else "disabled"
        else:
            self.parent.enable_merging = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Cluster merging has been {state}.")

    def toggle_debugging(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_debugging = not self.parent.enable_debugging
            state = "enabled" if self.parent.enable_debugging else "disabled"
        else:
            self.parent.enable_debugging = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Debugging has been {state}.")
        
    def toggle_evolving(self, enable=None, print_flag=False):
        
        if enable is None:
            self.parent.evolving = not self.parent.evolving
        else:
            self.parent.evolving = enable
        
        # Ensure that adding and merging are aligned with the evolving state
        self.parent.enable_adding = self.parent.evolving
        self.parent.enable_merging = self.parent.evolving

        # Print the new state
        state = "enabled" if self.parent.evolving else "disabled"
        if print_flag:
            print(f"Evolving has been {state}.")
