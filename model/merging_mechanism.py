
import torch
from utils.utils_plots import plot_clusters_with_kappa_pairs 

class MergingMechanism:
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
        self.merge_threshold = torch.exp(-((2*self.parent.num_sigma) ** 2)/self.parent.feature_dim)

    def find_pairs(self):
                
        self.kappa[self.kappa < self.parent.kappa_join] = 0
        cumulative_mask = torch.zeros_like(self.kappa, dtype=torch.bool)

        while True:
            first_max_indices = torch.argmax(self.kappa, dim=1)
            first_max_mask = torch.zeros_like(self.kappa, dtype=torch.bool)
            first_max_mask.scatter_(1, first_max_indices.unsqueeze(1), True)
            combined_mask = first_max_mask & first_max_mask.T
            upper_tri_mask = torch.triu(combined_mask, diagonal=1)

            if torch.sum(upper_tri_mask) == 0:
                break

            cumulative_mask |= upper_tri_mask
            matches = upper_tri_mask.nonzero(as_tuple=False)
            rows, cols = matches[:, 0], matches[:, 1]
            unique_nodes = torch.unique(torch.cat((rows, cols)))
            self.kappa[unique_nodes, :] = 0
            self.kappa[:, unique_nodes] = 0

        return cumulative_mask
    
    def perform_merge(self):

        if self.parent.debug_flag == 11: 
            plot_clusters_with_kappa_pairs(self, self.kappa, save=False)
    
        indices = torch.nonzero(self.find_pairs(), as_tuple=False)
        i_all, j_all = indices[:, 0], indices[:, 1]
        n_ij = self.parent.n[i_all] + self.parent.n[j_all]
        mu_ij = (self.parent.n[i_all, None] * self.parent.mu[i_all] + self.parent.n[j_all, None] * self.parent.mu[j_all]) / n_ij[:, None]
        mu_diff = self.parent.mu[i_all] - self.parent.mu[j_all]
        mu_diff_outer = torch.einsum('bi,bj->bij', mu_diff, mu_diff)
        S_ij = self.parent.S[i_all] + self.parent.S[j_all]
        S_ij += ((self.parent.n[i_all] * self.parent.n[j_all] / n_ij).unsqueeze(-1).unsqueeze(-1) * mu_diff_outer / self.parent.merging_weight)
        score_ij = (self.parent.n[i_all] * self.parent.score[i_all] + self.parent.n[j_all] * self.parent.score[j_all]) / n_ij
        num_pred_ij = self.parent.num_pred[i_all] + self.parent.num_pred[j_all]

        self.parent.mu[i_all] = mu_ij
        self.parent.S[i_all] = S_ij
        self.parent.n[i_all] = n_ij
        self.parent.age[i_all] = (self.parent.age[i_all] + self.parent.age[j_all]) / 2
        self.parent.score[i_all] = score_ij
        self.parent.num_pred[i_all] = num_pred_ij
        self.parent.Gamma[:, i_all] = torch.max(self.parent.Gamma[:, i_all], self.parent.Gamma[:, j_all])
        
        if 0:

            P_ij = self.parent.P[i_all] + self.parent.P[j_all]
            theta_ij = 2 * (self.parent.theta[i_all] * self.parent.theta[j_all]) / (self.parent.theta[i_all] + self.parent.theta[j_all])
            self.parent.P[i_all] = P_ij
            self.parent.theta[i_all] = theta_ij

        if self.parent.cond_cov:
            self.parent.S_inv[i_all] = torch.linalg.inv_ex(self.parent.S[i_all] / self.parent.n[i_all][:, None, None])[0]
        else:
            self.parent.S_inv[i_all] = torch.diag(1/torch.diag(self.parent.S_0))

        if torch.sum(self.parent.cluster_labels[i_all, 1:]) == 0 and torch.sum(self.parent.cluster_labels[j_all, 1:]) > 0:
            self.parent.cluster_labels[i_all] = self.parent.cluster_labels[j_all]

        self.parent.removal_mech.remove_clusters(j_all)
        indices_changed = torch.sort(torch.cat((i_all, j_all), dim=0))[0] #The removal mechanism shuffles the indices around
        indices_changed = indices_changed[indices_changed < self.parent.c]
        
        return indices_changed


    def batch_compute_volume(self):
        
        self.logdet = torch.full(self.valid_clusters.shape, float(0), device=self.parent.device)
        indices = self.valid_clusters.nonzero(as_tuple=True)
        num_indices = len(indices[0])

        if num_indices == 0:
            return  # nothing to merge
        
        n = self.parent.n[:self.parent.c]  # Shape: [num_clusters]
        mu = self.parent.mu[:self.parent.c][:, self.parent.feature_mask]  # Shape: [num_clusters, D]
        S = self.parent.S[:self.parent.c][:, self.parent.feature_mask][:, :, self.parent.feature_mask]  # Shape: [num_clusters, D, D]

        max_batch_size = 100000 / self.parent.feature_dim
        num_batches = torch.ceil(num_indices / max_batch_size +1) 
        batch_size = int(torch.ceil(num_indices / num_batches))
        if self.parent.cond_cov:
            self.logdet.diagonal().copy_(torch.linalg.slogdet(S / n[:, None, None])[1])
        else:
            self.logdet.diagonal().copy_(
                torch.sum(torch.log(torch.diagonal(S / n[:, None, None], dim1=-2, dim2=-1)), dim=1)
            )

        for batch_start in range(0, num_indices, batch_size):

            batch_end = min(batch_start + batch_size, num_indices)
            batch_indices_0 = indices[0][batch_start:batch_end]
            batch_indices_1 = indices[1][batch_start:batch_end]
            mu_diff = mu[batch_indices_0] - mu[batch_indices_1]  # Shape: [batch_size, D]
            mu_outer_product = mu_diff.unsqueeze(2) * mu_diff.unsqueeze(1)  # Shape: [batch_size, D, D]
            n_comb = n[batch_indices_0] + n[batch_indices_1]  # Shape: [batch_size]
            n_prod = n[batch_indices_0] * n[batch_indices_1]  # Shape: [batch_size]
            Sigma = S[batch_indices_0] + S[batch_indices_1]
            scalar_term = (n_prod / n_comb) / self.parent.merging_weight  # Shape: [batch_size]
            Sigma += scalar_term.unsqueeze(-1).unsqueeze(-1) * mu_outer_product  # Broadcasting to match shapes
            Sigma /= (n_comb - 1).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, D, D]

            if self.parent.cond_cov:
                V_batch = torch.linalg.slogdet(Sigma)[1]
            else:
                V_batch = torch.sum(torch.log(torch.diagonal(Sigma, dim1=-2, dim2=-1)), dim=-1)

            self.logdet[batch_indices_0, batch_indices_1] = V_batch
            self.logdet[batch_indices_1, batch_indices_0] = V_batch 

    def compute_kappa(self):
        
        log_det_sigma = self.logdet.diagonal()
        self.kappa = torch.exp(log_det_sigma.unsqueeze(0)-self.logdet) + torch.exp(log_det_sigma.unsqueeze(1)-self.logdet)
        self.kappa.fill_diagonal_(0)
        above_avg_indices = torch.exp(log_det_sigma-torch.sum(torch.log(torch.diag(self.parent.S_0)))) > 1
        self.kappa[above_avg_indices, :] = 0
        self.kappa[:, above_avg_indices] = 0


    def batch_merging_mechanism(self): 

        #A trick to find adjecent clusters with matmul
        co_occurrence_matrix = torch.zeros_like(self.parent.Gamma, device=self.parent.device, dtype=torch.bool)  # BxC
        co_occurrence_matrix = (self.parent.Gamma >= self.merge_threshold)*self.parent.matching.float()
        co_occurrence_matrix = co_occurrence_matrix.to(dtype=torch.float32) 
        valid_clusters = torch.matmul(co_occurrence_matrix.T, co_occurrence_matrix)
        valid_clusters = torch.triu(valid_clusters, diagonal=1)
        self.valid_clusters = valid_clusters.to(dtype=torch.bool)

        self.batch_compute_volume()
        self.compute_kappa()
        i_merged = self.perform_merge()
        
        return i_merged
