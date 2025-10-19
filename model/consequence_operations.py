import torch
import torch.nn as nn

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent

    def safe_argmax(self, one_hot_max_labels, dim):
        if one_hot_max_labels.size(dim) == 0:
            # Create an empty tensor with the same batch size and size 0 along the specified dimension
            new_shape = list(one_hot_max_labels.size())
            new_shape[dim] = 0
            return torch.empty(new_shape, dtype=torch.long, device=one_hot_max_labels.device)
        else:
            return torch.argmax(one_hot_max_labels, dim=dim)
        
    def defuzzify_batch(self, Z):
        
        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.compute_batched_normalized_gamma()
        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels[:,:,1:], dim=1)
    	
        # Find the indices of the maximum values in normalized_gamma along the cluster dimension
        max_indices = torch.argmax(normalized_gamma, dim=1)
        one_hot_max_labels = expanded_cluster_labels[torch.arange(normalized_gamma.shape[0]), max_indices, 1:]
        max_labels = self.safe_argmax(one_hot_max_labels, dim=1)

        return label_scores, max_labels

    def compute_batched_normalized_gamma(self):
        
        gamma_per_class = self.parent.Gamma[:, :self.parent.c].unsqueeze(2) * self.parent.cluster_labels[:self.parent.c, :].unsqueeze(0)  # Shape: B, C, M
        gamma_per_class = torch.nan_to_num(gamma_per_class, nan=0.0)
                
        sum_per_class = gamma_per_class.sum(dim=1, keepdim=True)  # Shape: B, 1, M
        sum_per_class = torch.nan_to_num(sum_per_class, nan=0.0)
        sum_per_class[sum_per_class == 0] = 1  # Avoid division by zero
        
        norm_gamma_per_class = (gamma_per_class / sum_per_class).sum(dim=2)
        norm_gamma_per_class[:,self.parent.cluster_labels[:self.parent.c, 0]==1] = 0 #Exclude the unlabelled clusters
        
        sum_norm = (norm_gamma_per_class * self.parent.Gamma[:, :self.parent.c]).sum(dim=1, keepdim=True)
        sum_norm = torch.nan_to_num(sum_norm, nan=0.0)
        sum_norm[sum_norm == 0] = 1  # Avoid division by zero
        
        norm_gamma = (norm_gamma_per_class * self.parent.Gamma[:, :self.parent.c]) / sum_norm
        
        return norm_gamma
