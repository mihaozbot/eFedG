import torch
#import torch.nn as nn
import matplotlib.pyplot as plt


class FederalOps:
    def __init__(self, parent):

        self.parent = parent
        self.thr_shift = self.parent.S_0_init[0,0]
        self.thr_shift_merge = torch.exp(-(1*torch.ones(1, device=self.parent.device))** 2/self.parent.feature_dim) #torch.exp(-((self.parent.num_sigma) ** 2)/self.feature_dim) #
    
    def federated_merging(self):
             
        self.parent.merging_mech.merge_threshold = torch.exp(-((2*self.parent.num_sigma) ** 2)/self.parent.feature_dim)
        
        with torch.no_grad():

            self.parent.clusterer.update_clustering_condition()
 
            cond_fed_merge = True
            while cond_fed_merge: #While merging happened, basically repeat until no merge happened after the loop
                cond_fed_merge = False
                ind_merged = torch.arange(self.parent.c)

                centers = self.parent.mu[ind_merged]
                #self.parent.feature_mask = (self.parent.fisher_scores > self.parent.kappa_features).to(self.parent.device) #torch.where(self.fisher_scores > self.kappa_features)[0]
                matching_unlabelled = (self.parent.cluster_labels[:self.parent.c, 0] == 1).unsqueeze(0).repeat(len(ind_merged), 1)  # Shape [self.parent.c, self.parent.c]
                labels = self.parent.cluster_labels[:self.parent.c]
                self.parent.matching = torch.matmul(labels, labels.T).bool()[ind_merged] + matching_unlabelled
                self.parent.mathematician.compute_batched_activation(centers)
 
                ind_merged = self.parent.merging_mech.batch_merging_mechanism()
                cond_fed_merge = (ind_merged.shape[0] != 0)


    def merge_model_statistics(self, model):

        n_fed = torch.sum(self.parent.n_glo)
        n_local = torch.sum(model.n_glo.clone())


        self.parent.n_glo = (self.parent.n_glo + model.n_glo.clone())
        self.parent.S_glo = (self.parent.S_glo + model.S_glo.clone()) + \
                            ((n_fed * n_local) / (n_fed + n_local)) * (self.parent.mu_glo - model.mu_glo.clone()) ** 2
        self.parent.mu_glo = (self.parent.mu_glo * n_fed + model.mu_glo.clone() * n_local) / (n_fed + n_local)
        self.parent.var_glo = self.parent.S_glo/(n_fed + n_local)

        self.parent.clusterer.update_S_0()
        if torch.isnan(self.parent.var_glo).any():
            print("Updated var_glo values in the FederalOps:", self.parent.var_glo[0])

        n_local = self.parent.n_glo.clone()
        n_fed = model.n_glo.clone()
        n_local_exp = n_local.unsqueeze(1)
        n_fed_exp = n_fed.unsqueeze(1)
        n_sum_exp = (n_local + n_fed).unsqueeze(1)
        self.parent.mu_cls = (self.parent.mu_cls * n_fed_exp + model.mu_cls * n_local_exp) / n_sum_exp
        mu_diff = self.parent.mu_cls - model.mu_cls
        scatter_update = (n_fed_exp * n_local_exp / n_sum_exp) * mu_diff ** 2
        self.parent.S_cls += model.S_cls + scatter_update

        self.parent.clusterer.compute_fisher_scores()


    def merge_model_privately(self, model, n_min, pred_min):

        with torch.no_grad(): 

            valid_clusters = (model.n[:model.c] > n_min)*(model.num_pred[:model.c] >= pred_min)
            num_valid_clusters = valid_clusters.sum()
            valid_indices = torch.where(valid_clusters)[0]
            self.merge_model_statistics(model)
            self.parent.clusterer.update_clustering_condition()
            matching_unlabelled = (model.cluster_labels[:model.c][valid_clusters, 0] == 1).unsqueeze(0).repeat(num_valid_clusters, 1) 
            self.parent.matching = matching_unlabelled
            self.parent.mathematician.compute_batched_activation(model.mu[:model.c][valid_clusters])

            if self.parent.Gamma.numel() > 0:
                max_values, max_indices = torch.max(self.parent.Gamma, dim=1)
                to_increment = max_values >= model.clusterer.Gamma_max
                self.parent.clusterer.batch_increment_clusters_with_covariance(model.mu[:model.c][valid_clusters][to_increment], 
                                                                            model.cluster_labels[:model.c][valid_clusters][to_increment], 
                                                                            model.n[:model.c][valid_clusters][to_increment], 
                                                                            max_indices[to_increment],
                                                                            model.S[:model.c][valid_clusters][to_increment])
                
                valid_clusters[valid_clusters.clone()] = ~to_increment.clone()
                num_valid_clusters = valid_clusters.sum()
                valid_indices = torch.where(valid_clusters)[0]

            before_size = self.parent.c
            after_size = self.parent.c + num_valid_clusters
            self.parent.overseer.ensure_capacity(after_size+2)  # Ensure there's enough space

            new_indices = torch.arange(before_size, before_size + len(valid_indices), device = model.mu.device)
            self.parent.mu.data[new_indices] = model.mu.data[valid_indices].clone()
            self.parent.mu_og.data[new_indices] = model.mu_og.data[valid_indices].clone()
            self.parent.S.data[new_indices] = model.S.data[valid_indices].clone()
            self.parent.n.data[new_indices] = model.n.data[valid_indices].clone()
            self.parent.S_inv.data[new_indices] = model.S_inv.data[valid_indices].clone()

            #Copy the concequence 
            if 0:
                self.parent.P.data[new_indices] = model.P.data[valid_indices]
                self.parent.theta.data[new_indices] = model.theta.data[valid_indices]
                    
            self.parent.cluster_labels[new_indices] = model.cluster_labels[valid_indices].clone()
            self.parent.age[new_indices] = model.age[valid_indices].clone()
            self.parent.score[new_indices] = model.score[valid_indices].clone()
            self.parent.num_pred[new_indices] = model.num_pred[valid_indices].clone()

        self.parent.Gamma = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=False)
                
        self.parent.c = after_size  
        #self.parent.cond_cov = self.parent.feature_dim < 10 # self.parent.c*self.parent.feature_dim*(self.parent.feature_dim +1)/2  < torch.sum(self.parent.n_glo)
    
    def remove_irrelevant_clusters(self):

        S = self.parent.S[:self.parent.c][:, self.parent.feature_mask][:, :, self.parent.feature_mask]
        importance = torch.sqrt(torch.exp(torch.linalg.slogdet(S)[1]))
        self.parent.importance = (importance ) / (torch.sum(importance))
        sorted_importance, sorted_indices = torch.sort(self.parent.importance, descending=True)
        cumulative_importance = torch.cumsum(sorted_importance, dim=0)
        threshold_indices = (cumulative_importance > self.parent.thr_relevance).nonzero(as_tuple=True)[0]


        if self.parent.debug_flag in [16]:

            if len(threshold_indices) > 0:
                cutoff_point = threshold_indices[0]

            # Plot the sorted importance and its cumulative sum
            plt.figure(figsize=(6, 4))
            plt.plot(sorted_importance.numpy(), marker='o', label='Importance')
            plt.plot(cumulative_importance.numpy(), marker='o', linestyle='--', label='Cumulative Importance')
            if len(threshold_indices) > 0:
                plt.axvline(x=cutoff_point, color='r', linestyle='--', label=f'Cutoff at {self.parent.thr_relevance*100}% Importance')
            plt.title('Sorted Importance and Cumulative Importance of Clusters')
            plt.xlabel('Sorted Cluster Index')
            plt.ylabel('Importance')
            plt.grid(True)
            plt.legend()
            plt.show()
            


        if len(threshold_indices) > 0:
            first_excess_index = threshold_indices[0]
            indices_to_remove = sorted_indices[first_excess_index + 1:] 
            self.parent.removal_mech.remove_clusters(indices_to_remove)