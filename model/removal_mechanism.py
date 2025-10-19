import torch

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent
        self.push_th = (self.parent.num_sigma)**2

    def batch_update_scores(self, Z, y):

        normalized_gammas = self.parent.consequence.compute_batched_normalized_gamma() #[B, c]

        if self.parent.class_type == 0:
            correct_clusters_mask = self.parent.cluster_labels[:self.parent.c, y].bool()
            score_increment = torch.sum(normalized_gammas.T*correct_clusters_mask, dim=1)
            gamma_increment = torch.sum(normalized_gammas.T, dim=1)
            
        elif self.parent.class_type == 1:
            _, predicted_labels = self.parent.consequence.defuzzify_batch(Z)
            correct_clusters_mask = predicted_labels == y
            score_increment = torch.sum(normalized_gammas.T*correct_clusters_mask, dim=1)
            gamma_increment = torch.sum(normalized_gammas.T, dim=1)
            
        self.parent.score[:self.parent.c] = (self.parent.num_pred[:self.parent.c] * self.parent.score[:self.parent.c] + score_increment) / (self.parent.num_pred[:self.parent.c] + gamma_increment) #.type(torch.float32)
        self.parent.num_pred[:self.parent.c] += gamma_increment

        if self.parent.debug_flag  == 6:
            if torch.isnan(gamma_increment).any():
                print("Error score update")

    def remove_irrelevant(self, c_max):
        if self.parent.c < 2:
            return
        
        num_clusters_to_remove = len(self.parent.matching) - c_max
        if num_clusters_to_remove > 0:

            clusters_eligible_for_removal = [i for i in self.parent.matching if self.parent.num_pred[i] > 1]
            sorted_indices = sorted(clusters_eligible_for_removal, key=lambda i: (self.parent.num_pred[i], -self.parent.score[i]))
            indices_to_remove = sorted(sorted_indices, reverse=True)
            indices_to_remove = indices_to_remove[:num_clusters_to_remove]
            
            with torch.no_grad():
      
                for index in indices_to_remove:
                    self.remove_clusters(index)
                    
    def remove_clusters(self, indices_to_remove):        

        # Replace the data of clusters to be removed with the data from the end
        self.parent.c  = self.parent.c - indices_to_remove.shape[0]
        indices_to_remove_filtered = indices_to_remove[indices_to_remove < self.parent.c ]
        valid_indices = torch.ones(self.parent.c + indices_to_remove.shape[0], dtype=torch.bool, device=self.parent.device)
        valid_indices[indices_to_remove] = 0
        remaining_indices = torch.nonzero(valid_indices).squeeze(1)
        replacement_indices = remaining_indices[remaining_indices >= self.parent.c]


        self.parent.mu.data[indices_to_remove_filtered] = self.parent.mu.data[replacement_indices]
        self.parent.mu_og.data[indices_to_remove_filtered] = self.parent.mu_og.data[replacement_indices]
        self.parent.n.data[indices_to_remove_filtered] = self.parent.n.data[replacement_indices]
        self.parent.S.data[indices_to_remove_filtered] = self.parent.S.data[replacement_indices]
        self.parent.S_inv.data[indices_to_remove_filtered] = self.parent.S_inv.data[replacement_indices]
        self.parent.age[indices_to_remove_filtered] = self.parent.age[replacement_indices]
        self.parent.score[indices_to_remove_filtered] = self.parent.score[replacement_indices]
        self.parent.num_pred[indices_to_remove_filtered] = self.parent.num_pred[replacement_indices]
        self.parent.cluster_labels[indices_to_remove_filtered] = self.parent.cluster_labels[replacement_indices]

        if 0:
            self.parent.P.data[indices_to_remove_filtered] = self.parent.P.data[replacement_indices]
            self.parent.theta.data[indices_to_remove_filtered] = self.parent.theta.data[replacement_indices]

        self.parent.Gamma[:, indices_to_remove_filtered] = self.parent.Gamma[:, replacement_indices]
        self.parent.Gamma = self.parent.Gamma[:, :self.parent.c] #torch.cat((self.parent.Gamma[:,:cluster_index], self.parent.Gamma[:,cluster_index+1:]), dim=1)

        return indices_to_remove_filtered
    