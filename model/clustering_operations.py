import torch
import matplotlib.pyplot as plt

class ClusteringOps:
    def __init__(self, parent):
        self.parent = parent # Reference to the parent class
        self.feature_dim = parent.feature_dim # Number of features in the dataset
        self.min_d2 = ((self.parent.num_sigma)**2)
        self.Gamma_max = torch.exp(-self.min_d2/self.parent.feature_dim) # Maximum value of Gamma (used to determine if a new cluster should be added)

    def kmeans_plusplus_init(self, X, n_clusters, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)
        
        n_samples, _ = X.shape
        device = X.device
        centroids = torch.empty((n_clusters, X.shape[1]), dtype=X.dtype, device=device)
        centroid_indices = torch.empty(n_clusters, dtype=torch.long, device=device)

        initial_index = torch.randint(n_samples, (1,), device=device)
        centroids[0] = X[initial_index]
        centroid_indices[0] = initial_index

        S0_diag = torch.diag(self.parent.S_0) 
        distances = torch.sum(
            ((X - centroids[0]) ** 2) / S0_diag, dim=1
        )  # Shape (n_samples,)
        
        for i in range(1, n_clusters):
    
            probabilities = distances.clone()
            probabilities /= probabilities.sum()

            next_centroid_index = torch.argmax(probabilities)
            centroids[i] = X[next_centroid_index]
            centroid_indices[i] = next_centroid_index

            dist_to_new_centroid = torch.sum(
                ((X - centroids[i]) ** 2) / S0_diag, dim=1
            )
            
            distances = torch.minimum(distances, dist_to_new_centroid)

        return centroids, centroid_indices
        
    def reduce_by_label(self, Z, n, y):
        unique_labels = torch.unique(y)
        for label in unique_labels:
            Z_sub = Z[y == label]
            n_sub = n[y == label]
            self.add_new_clusters(Z_sub, n_sub)
        
    def add_new_clusters(self, Z, n):
      
        while Z.shape[0] > self.parent.N_r*torch.sqrt(self.parent.feature_dim): 

            num_clusters = torch.ceil(torch.min(self.parent.N_r*torch.sqrt(self.parent.feature_dim))).to(torch.int32)

            sample_centers, _ = self.kmeans_plusplus_init(Z, num_clusters)
            new_n = torch.ones(num_clusters, device=self.parent.device)
            labels = torch.zeros(num_clusters, dtype=torch.long, device=self.parent.device)
            cluster_indices = torch.arange(self.parent.c, self.parent.c + num_clusters, device=self.parent.device)
  
            self.batch_add_cluster(sample_centers, labels, new_n)

            d2 = torch.sum(((Z.unsqueeze(1) - sample_centers) ** 2) / torch.diag(self.parent.S_0), dim=2)
            min_d2, min_indices = torch.min(d2, dim=1)
            
            closest_mask = min_d2 <= (self.min_d2)
            jm = cluster_indices[min_indices[closest_mask]]

            self.batch_increment_clusters(Z[closest_mask], labels[min_indices[closest_mask]], n[closest_mask], jm)

            Z = Z[~closest_mask]
            n = n[~closest_mask]
                
        new_n = torch.ones(Z.shape[0], device=self.parent.device)
        labels = torch.zeros(Z.shape[0], dtype=torch.long, device=self.parent.device)
        self.batch_add_cluster(Z, labels, new_n)    
        return

    def update_S_0(self):
        self.parent.S_0 = torch.clamp(torch.diag(self.parent.var_glo/(self.parent.N_r)), min=self.parent.S_0_init)

    def batch_add_cluster(self, Z, y, n, S=None):

        num_added = Z.shape[0]
        self.parent.overseer.ensure_capacity(self.parent.c+num_added+2)
        self.parent.mu[self.parent.c:(self.parent.c+num_added)] = Z
        self.parent.mu_og[self.parent.c:(self.parent.c+num_added)] = Z

        if S== None:
            self.parent.S[self.parent.c:(self.parent.c+num_added)] = n[:, None, None]*self.parent.S_0
        self.parent.n[self.parent.c:(self.parent.c+num_added)] = n
        
        self.parent.score[self.parent.c:(self.parent.c+num_added)] = 1
        self.parent.num_pred[self.parent.c:(self.parent.c+num_added)] = 1
        self.parent.age[self.parent.c:(self.parent.c+num_added)] = 1

        self.parent.S_inv[self.parent.c:(self.parent.c+num_added)] = (1.0 / (self.parent.S_0.diagonal()))* torch.eye(self.parent.feature_dim, device=self.parent.device).unsqueeze(0)
        #self.parent.theta[self.parent.c:(self.parent.c+num_added),0] = self.parent.one_hot_labels[y]
                
        if self.parent.c == 0:
            self.parent.Gamma = torch.zeros(Z.shape[0], num_added, device=self.parent.device)
        else:
            expanded_Gamma = torch.empty(self.parent.Gamma.shape[0], self.parent.c + num_added, device=self.parent.device)
            expanded_Gamma[:, :self.parent.c] = self.parent.Gamma
            expanded_Gamma[:, self.parent.c:] = 0
            self.parent.Gamma = expanded_Gamma

        self.parent.cluster_labels[self.parent.c:(self.parent.c+num_added)] = self.parent.one_hot_labels[y]
        self.parent.c += num_added
     
    def batch_update_clusters(self, Z, y, n):
        
        if self.parent.Gamma.numel() > 0:
            max_values, max_indices = torch.max(self.parent.Gamma, dim=1)
            to_increment = max_values >= self.Gamma_max
            self.batch_increment_clusters(Z[to_increment], y[to_increment], n[to_increment], max_indices[to_increment])

        if self.parent.c == 0:
            to_increment = torch.zeros_like(n, device=self.parent.device) > 1
        
        to_add_unrepresented = to_increment == False
        if torch.any(to_add_unrepresented):
            
            c_pre = self.parent.c.detach().clone()
            self.reduce_by_label(Z[to_add_unrepresented], n[to_add_unrepresented], y[to_add_unrepresented])

            if self.parent.debug_flag == 7:
                
                print("Shape of Z[to_add_unrepresented, 0]:", Z[to_add_unrepresented, 0].shape)
                print("Shape of new_cluster_centers:", self.parent.mu[c_pre:self.parent.c, 0].shape)
                print("Maximum count in clusters:", torch.max(self.parent.n[:self.parent.c]))

                plt.scatter(
                    Z[to_add_unrepresented, 0].cpu().numpy(),
                    Z[to_add_unrepresented, 1].cpu().numpy(),
                    c='blue',
                    label='Before reduce_by_label'
                )
                plt.scatter(
                    self.parent.mu[c_pre:self.parent.c, 0].cpu().numpy(),
                    self.parent.mu[c_pre:self.parent.c, 1].cpu().numpy(),
                    c='red',
                    s=100,
                    label='New Cluster Centers'
                )

                plt.title('Before and After reduce_by_label')
                plt.legend()
                plt.show()

    def batch_increment_clusters(self, Z, y, n, jm, decay = 0.9):

        # decay 1 for no decay

        c = self.parent.c
        dtype, device = Z.dtype, Z.device
        eps = torch.finfo(dtype).eps

        affected = torch.unique(jm)      
        decay_vec = torch.ones(c, dtype=dtype, device=device) 
        decay_vec[affected] = decay   

        num_pre = self.parent.n[:c].clone().to(dtype)
        num_pre_decay = num_pre * decay_vec
        num_added = torch.bincount(jm, weights=n.to(dtype), minlength=c).to(dtype)
        num_post = num_pre_decay + num_added
        one_hot_jm = torch.nn.functional.one_hot(jm, num_classes=c).to(dtype)   # (B,K)
        weighted_one_hot_jm = one_hot_jm * n.unsqueeze(1)                        # (B,K)

        mu_pre = self.parent.mu[:c]
        denom = torch.clamp(num_post, min=eps).unsqueeze(1)                      # (K,1)
        self.parent.mu[:c] = (weighted_one_hot_jm.T @ Z +
                            num_pre_decay.unsqueeze(1) * mu_pre) / denom
        self.parent.n[:c] = num_post

        if self.parent.cond_cov:

            # full covariance
            Z_diff = self.parent.mu[jm] - Z
            Z_diff_weighted = Z_diff * n.unsqueeze(1)
            outer_products = torch.einsum('bi,bj->bij', Z_diff_weighted, Z_diff)
            S_update = torch.einsum('bi,bjk->ijk', one_hot_jm, outer_products)   # (K,D,D)

            # decay only affected clusters' scatter, then floor at S_0
            S_pre = self.parent.S[:c]
            S_pre = decay_vec.view(-1, 1, 1) * S_pre
            self.parent.S[:c] = torch.maximum(self.parent.S_0, S_pre) + S_update

            cov = self.parent.S[:c] / denom.unsqueeze(2)
            self.parent.S_inv[:c] = torch.linalg.inv_ex(cov).inverse

        else:

            # diagonal-only variance
            S_prev = self.parent.S[:c]
            S_prev_diag = torch.diagonal(S_prev, dim1=1, dim2=2)                 # (K,D)
            S0_diag = torch.diag(self.parent.S_0).to(dtype)                       # (D,)
            S_prev_diag = torch.maximum(S0_diag, S_prev_diag * decay_vec.unsqueeze(1))

            Z_diff = self.parent.mu[jm] - Z
            sq = (Z_diff ** 2) * n.unsqueeze(1)                                   # (B,D)
            S_update_diag = one_hot_jm.T @ sq                                     # (K,D)

            S_diag_new = S_prev_diag + S_update_diag
            self.parent.S[:c] = torch.diag_embed(S_diag_new)

            cov_diag = S_diag_new / torch.clamp(num_post, min=eps).unsqueeze(1)
            self.parent.S_inv[:c] = torch.diag_embed(1.0 / torch.clamp(cov_diag, min=eps))


    def batch_increment_clusters_with_covariance(self, mu, y, n, jm, S):
        #This is used to udpate the clusters with other clusters directly

        num_pre = self.parent.n[:self.parent.c].clone()  # Shape: (num_clusters,)
        num_added = torch.bincount(jm, weights=n, minlength=num_pre.size(0)).to(dtype=mu.dtype)
        num_post = num_pre + num_added

        one_hot_jm = torch.nn.functional.one_hot(jm, num_classes=self.parent.c).to(mu.dtype)  # Shape: (batch_size, num_clusters)
        weighted_one_hot_jm = one_hot_jm * n.unsqueeze(1)  # Shape: (batch_size, num_clusters)

        self.parent.mu[:self.parent.c] = (torch.matmul(weighted_one_hot_jm.T, mu) + 
                                        num_pre.unsqueeze(1) * self.parent.mu[:self.parent.c]) / num_post.unsqueeze(1)
        self.parent.n[:self.parent.c] = num_post

        mu_diff = self.parent.mu[jm] - mu   # Shape: (num_clusters, D)
        mu_diff_outer = torch.einsum('bi,bj->bij', mu_diff, mu_diff) 
        mean_diff_weight = (num_pre[jm] * n / num_post[jm]).unsqueeze(-1).unsqueeze(-1) 
        cov_update_batch = S + mean_diff_weight * mu_diff_outer / self.parent.merging_weight  # Shape: (batch_size, D, D)
        S_update = torch.einsum('bi,bjk->ijk', one_hot_jm, cov_update_batch)
        self.parent.S[:self.parent.c] += S_update
        self.parent.S_inv[:self.parent.c] = torch.linalg.inv_ex(
            self.parent.S[:self.parent.c] / num_post.unsqueeze(1).unsqueeze(2)
        ).inverse

    def update_global_statistics(self, z, label):

        self.parent.n_glo[label] += 1
        total_samples = torch.sum(self.parent.n_glo)
        e_glo = z - self.parent.mu_glo 
        self.parent.mu_glo += e_glo / total_samples  
        self.parent.S_glo = self.parent.S_glo + (total_samples - 1) / total_samples * e_glo *(z -  self.parent.mu_glo)
        self.parent.var_glo = self.parent.S_glo / total_samples 

        self.update_S_0() 
        
    def batch_update_global_statistics(self, Z, y):

        one_hot_encoded = self.parent.one_hot_labels[y]
        counts = one_hot_encoded.sum(dim=0)

        num_pre = self.parent.n_glo.clone()

        self.parent.n_glo[:self.parent.num_classes] += counts
        self.parent.mu_glo = (torch.sum(Z, dim = 0) + torch.sum(num_pre)*self.parent.mu_glo)/ torch.sum(self.parent.n_glo)

        Z_centered = Z - self.parent.mu_glo
        self.parent.S_glo += (Z_centered ** 2).sum(dim=0)
        self.parent.var_glo = self.parent.S_glo / torch.sum(self.parent.n_glo)

        self.update_S_0()
        
        one_hot_labels = torch.nn.functional.one_hot(y, num_classes=self.parent.num_classes).float()
        Z_sum_per_class = torch.matmul(one_hot_labels.T, Z)
        self.parent.mu_cls = (num_pre.unsqueeze(1) * self.parent.mu_cls + Z_sum_per_class) / self.parent.n_glo.unsqueeze(1)
        Z_diff = Z.unsqueeze(0) - self.parent.mu_cls.unsqueeze(1)
        Z_diff_squared = (Z_diff ** 2) * one_hot_labels.T.unsqueeze(2)
        self.parent.S_cls += Z_diff_squared.sum(dim=1)

        if self.parent.num_classes>1:
            self.compute_fisher_scores()
        if self.parent.c > 1 and self.parent.num_classes==1:
            self.compute_fisher_unsupervised()
        else:
            self.parent.feature_mask = torch.ones(self.parent.feature_dim, dtype=torch.bool, device=self.parent.device)

    def compute_fisher_unsupervised(self):

        diff = self.parent.mu[:self.parent.c] - self.parent.mu_glo.unsqueeze(0)
        fisher_num = torch.sum(self.parent.n[:self.parent.c].unsqueeze(1) * (diff ** 2), dim=0)
        fisher_den = torch.sum(
            torch.diagonal(self.parent.S[:self.parent.c], dim1=1, dim2=2) / self.parent.n[:self.parent.c].unsqueeze(1), dim=0
        )  # Shape: (D,)
        self.parent.fisher_scores = fisher_num / fisher_den
        self.parent.fisher_scores = self.parent.fisher_scores / torch.max(self.parent.fisher_scores)
        self.parent.feature_mask = (self.parent.fisher_scores > self.parent.kappa_features).to(self.parent.device)
        #self.parent.num_sigma = torch.sqrt(torch.sum(self.parent.feature_mask)).to(self.parent.device)

    def compute_fisher_scores(self):

        fisher_num = torch.zeros((self.parent.mu_cls.shape[1]), device=self.parent.device)
        fisher_den = torch.zeros_like(fisher_num, device=self.parent.device)
        
 
        for label in range(self.parent.num_classes):
                        
            fisher_num += self.parent.n_glo[label] * (self.parent.mu_cls[label] - self.parent.mu_glo ) ** 2
            fisher_den += self.parent.S_cls[label] #/self.parent.S_glo

        
        diff = self.parent.mu_cls - self.parent.mu_glo.unsqueeze(0) 
        self.parent.fisher_scores =  torch.sum(self.parent.n_glo.unsqueeze(1) * (diff ** 2), dim=0) / torch.sum(self.parent.S_cls, dim=0) 
        self.parent.fisher_scores = (self.parent.fisher_scores)/(torch.max(self.parent.fisher_scores)).to(self.parent.device)
        
        self.parent.feature_mask = (self.parent.fisher_scores > self.parent.kappa_features).to(self.parent.device)
        #self.parent.num_sigma = torch.sqrt(torch.sum(self.parent.feature_mask)).to(self.parent.device)

    def update_clustering_condition(self):
    
        self.parent.cond_cov = self.parent.feature_dim < 10 #self.parent.c*self.parent.feature_dim*self.parent.feature_dim <= torch.sum(self.parent.n) #self.parent.feature_dim < 10# 
