
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_mean_std(metrics, key):

    means = []
    stds = []
    num_rounds = len(metrics)
    for round_idx in range(num_rounds):
        round_values = [client_metric['binary'][key] for client_metric in metrics[round_idx]['client_metrics']]
        means.append(np.mean(round_values))
        stds.append(np.std(round_values))

    return means, stds


def calculate_metrics(pred_max, test_dataset):
    _, test_labels = test_dataset.tensors

    if max(test_labels)==1:
        weight = "binary"
    else:
        weight = "weighted"
        
    accuracy = accuracy_score(test_labels, pred_max)
    precision = precision_score(test_labels, pred_max, average=weight, zero_division='warn')
    recall = recall_score(test_labels, pred_max, average=weight, zero_division='warn')
    f1 = f1_score(test_labels, pred_max, average=weight, zero_division='warn')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def calculate_roc_auc(outputs, test_dataset, plot_flag=False):
 
    _, test_labels = test_dataset.tensors
    outputs = torch.softmax(outputs.cpu(),dim=1)  # Exclude the first column related to class 0
    test_labels = test_labels - 1  # Adjust labels to be 0, 1, 2 instead of 1, 2, 3
            
    if  len(np.unique(test_labels)) == 2:
        weight = "None"
    else:
        weight = "weighted"

    if len(np.unique(test_labels)) == 2:
        positive_class_scores = outputs[:, 1]
        roc_auc = roc_auc_score(test_labels, positive_class_scores)
    else:
        # Multiclass classification case
        roc_auc = roc_auc_score(test_labels, outputs, multi_class='ovr', average=weight)

    if plot_flag:
    
        fpr, tpr, _ = roc_curve(test_labels, positive_class_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.1f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc
    
def calculate_metrics_statistics(metrics_list, resolution=2):

    flat_list = [item for sublist in metrics_list for item in sublist]
    keys = flat_list[0].keys()
    mean_metrics = {key: np.mean([metrics[key] for metrics in flat_list]) for key in keys}
    std_metrics = {key: np.std([metrics[key] for metrics in flat_list], ddof=1) for key in keys}

    return format_metrics(mean_metrics, std_metrics, resolution)

def format_metrics(mean_metrics, std_metrics, resolution):
    formatted = {key: f"{mean_metrics[key]:.{resolution}f}" + r"{±}" + f"{std_metrics[key]:.{resolution}f}" for key in mean_metrics}
    return formatted

def calculate_cluster_stats(client_clusters):

    all_clusters = [cluster for client in client_clusters for cluster in client]
    avg_clusters = np.mean(all_clusters)
    std_clusters = np.std(all_clusters, ddof=1)

    return avg_clusters, std_clusters

def calculate_unsupervised_metrics(assignments, dataset):
  
    data, labels = dataset

    if len(np.unique(assignments))>1:
        silhouette = silhouette_score(data, assignments, metric="mahalanobis")
        davies_bouldin = davies_bouldin_score(data, assignments)
        calinski_harabasz = calinski_harabasz_score(data, assignments)

        # Note: The following metrics require true labels to be meaningful.
        # They are included here for completeness, but should be used only if true labels are available.
        adjusted_rand = adjusted_rand_score(labels, assignments)
        normalized_mutual_info = normalized_mutual_info_score(labels, assignments)
        homogeneity = homogeneity_score(labels, assignments)
        completeness = completeness_score(labels, assignments)
        v_measure = v_measure_score(labels, assignments)
        
    else:
        silhouette = 0
        davies_bouldin = 0
        calinski_harabasz = 0
        adjusted_rand = 0
        normalized_mutual_info = 0
        homogeneity = 0
        completeness = 0
        v_measure = 0

    return {
        "silhouette_score": silhouette,
        #"davies_bouldin_score": davies_bouldin,
        #"calinski_harabasz_score": calinski_harabasz,
        "adjusted_rand_score": adjusted_rand,
        "normalized_mutual_info_score": normalized_mutual_info,
        #"homogeneity_score": homogeneity,
        #"completeness_score": completeness,
        "v_measure_score": v_measure
    }

