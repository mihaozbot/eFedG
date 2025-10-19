import torch
import torch.nn as nn
from model.clustering_operations import ClusteringOps
from model.removal_mechanism import RemovalMechanism 
from model.merging_mechanism import MergingMechanism
from model.math_operations import MathOps
from model.consequence_operations import ConsequenceOps
from model.model_operations import ModelOps
from model.federated_operations import FederalOps


class eFedG(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, kappa_n, num_sigma, kappa_join, N_r, c_max, kappa_features, device, thr_relevance):
        super(eFedG, self).__init__()
        self.device = device
        
        # Convert parameters to tensors if they are numerical values
        self.feature_dim = torch.tensor(feature_dim, device=self.device)
        self.kappa_n = torch.tensor(kappa_n, device=self.device)
        self.num_sigma = torch.tensor(num_sigma, device=self.device)
        self.kappa_join = torch.tensor(kappa_join, device=self.device)
        
        # Use tensors for matrices and vectors
        self.S_0 = 10**(-64/torch.sqrt(self.feature_dim))*torch.eye(self.feature_dim.item(), device=self.device)  # eye requires an integer
        self.S_0_init = self.S_0.clone()
        self.N_r = torch.tensor(N_r, device=self.device)
        self.num_classes = torch.tensor(num_classes, device=self.device)
        self.c_max = torch.tensor(c_max, device=self.device)
        #self.order = torch.tensor(1, device=self.device)
        self.thr_relevance = torch.tensor(thr_relevance, device=self.device)
        #self.n_phi = self.order * self.feature_dim + 1
        #self.class_type = torch.tensor(0, device=self.device)
        self.kappa_features = torch.tensor(kappa_features, device=self.device)
        self.merging_weight = torch.tensor(1, device=self.device)
        #self.epsilon = torch.tensor(1e-8, device=self.device)
        #self.scale_unlabeled = torch.tensor((1/2)**2, device=self.device)
        #self.student_factor = torch.tensor(1, device=self.device)
        
        # Dynamic properties initialized with tensors
        self.c = torch.tensor(0, dtype=torch.int16, device=self.device) # Number of active clusters
        self.Gamma = torch.zeros(0, dtype=torch.float32, device=device, requires_grad=False)
        self.current_capacity = c_max #Initialize current capacity, which will be expanded as needed during training 
        self.cluster_labels = torch.zeros((self.current_capacity, num_classes), dtype=torch.float32, device=device) #Initialize cluster labels
        
        #Support tensors
        self.score = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster scores
        self.num_pred = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device) #Initializs number of predictions
        self.age = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster age
        self.one_hot_labels = torch.eye(num_classes, dtype=torch.float16, device=device) #One hot labels 
        self.feature_mask = torch.ones(feature_dim, dtype=torch.bool, device=device) 

        # Trainable parameters
        self.n = (torch.zeros(self.current_capacity, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster sizes
        self.mu = (torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster means
        self.mu_og = (torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster means
        self.S = (torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device, requires_grad=False)) # Initialize covariance matrices       
        self.S_inv = torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device, requires_grad=False) # Initialize covariance matrices

        #Consequence ARX local linear models 
        if 0:
            self.P0 = 1e1*torch.eye(self.n_phi, dtype=torch.float32, device=device, requires_grad=False)
            self.P = (torch.zeros(self.current_capacity, self.n_phi, self.n_phi, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
            self.theta = nn.Parameter(torch.zeros(self.current_capacity, self.n_phi, self.num_classes, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        
        # Global statistics
        self.n_glo = torch.ones((num_classes), dtype=torch.float32, device=device)  # Global number of sampels per class
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Global mean
        self.S_glo = 10**(-64/torch.sqrt(self.feature_dim))*torch.ones((feature_dim), dtype=torch.float32, device=device) # Sum of squares for global variance
        self.mu_cls = torch.zeros((num_classes, feature_dim), dtype=torch.float32, device=device)  # Mean per class
        self.S_cls = 10**(-64/torch.sqrt(self.feature_dim))*torch.ones((num_classes, feature_dim), dtype=torch.float32, device=device)  # Sum of squares for variance per class

        # Initialize subclasses
        self.overseer = ModelOps(self)
        self.mathematician = MathOps(self)
        self.clusterer = ClusteringOps(self)
        self.merging_mech = MergingMechanism(self)
        self.removal_mech = RemovalMechanism(self)
        self.consequence = ConsequenceOps(self)
        self.federal_agent = FederalOps(self)

    def set_debugging_flag(self, debug_flag):
        self.debug_flag = debug_flag

    def federated_merging(self):
        self.federal_agent.federated_merging()

    @torch.no_grad()    
    def batch_clustering(self, Z, y):

        self.clusterer.update_clustering_condition()
        self.clusterer.batch_update_global_statistics(Z, y)
        self.fisher_scores = torch.ones(self.feature_dim, dtype=torch.bool, device=self.device)
        self.mathematician.compute_batched_activation(Z)
        self.clusterer.batch_update_clusters(Z, y, torch.ones(len(Z), device=self.device))
        
        if 0:
            self.matching = (self.n[:self.c]>self.kappa_n)
            self.merging_mech.batch_merging_mechanism()

    def forward(self, Z):
         
        self.mathematician.compute_batched_activation(Z)
        soft_scores, preds_max = self.consequence.defuzzify_batch(Z) 
        clusters = self.Gamma.argmax(dim=1)

        return soft_scores, preds_max, clusters
    