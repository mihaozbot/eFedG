import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from itertools import combinations
from math import comb
from utils.utils_metrics import calculate_mean_std
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
import matplotlib.font_manager as fm
from matplotlib.markers import MarkerStyle
import matplotlib.font_manager as fm
from matplotlib.transforms import Bbox


def plot_confusion_matrix(pred_max, test_dataset):

    _, test_labels = test_dataset.tensors
    num_classes = len(np.unique(test_labels))
    class_names = [str(i) for i in range(num_classes)]
    cm = confusion_matrix(test_labels, pred_max)
    fig_width = 4
    fig_height = 3
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_dataset_split(client_data, test_dataset):

    _, y_test = test_dataset

    num_clients = len(client_data)
    classes, _ = np.unique(y_test.numpy(), return_counts=True)
    num_classes = len(classes)

    client_counts = np.zeros((num_clients, num_classes))
    test_counts = np.zeros(num_classes)

    for i, (_, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client.numpy(), return_counts=True)
        for class_label, count in zip(unique, counts):
            client_counts[i, class_label] = count

    unique, counts = np.unique(y_test.numpy(), return_counts=True)
    for class_label, count in zip(unique, counts):
        test_counts[class_label] = count

    labels = [f'Client {i+1}' for i in range(num_clients)] + ['Test Set']

    fig_width = 5
    fig_height = 3

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_width = 0.5
    opacity = 0.8

    bottom = np.zeros(num_clients + 1)

    for class_label in range(num_classes):
        bar_values = np.append(client_counts[:, class_label], test_counts[class_label])
        bars = ax.bar(np.arange(num_clients + 1), bar_values, bar_width, alpha=opacity, label=f'Class {class_label+1}', bottom=bottom)

        for bar_index, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, bottom[bar_index] + height / 2),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='center')

        bottom += bar_values

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(np.arange(num_clients + 1))
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    return fig

def plot_with_intervals(rounds, means, stds, label, color, y_label, alpha=1.0, marker='o'):
    plt.errorbar(rounds, means, yerr=stds, fmt=marker + '-', ecolor=color, elinewidth=3, capsize=0, label=label, color=color, alpha=alpha)
    lower_bound = np.array(means) - np.array(stds)
    upper_bound = np.array(means) + np.array(stds)
    plt.fill_between(rounds, lower_bound, upper_bound, color=color, alpha=alpha * 0.15)
    plt.xlabel('Communication Round')
    plt.ylabel(y_label)

    plt.grid(True)

def plot_model_metric(rounds, values, label, color, linestyle, alpha=1.0, marker='o'):
    plt.plot(rounds, values, marker + linestyle, label=label, color=color, linewidth=2, alpha=alpha)

def plot_metric_data(metrics, metric_keys, rounds, title, legend= True):
    client_color = plt.get_cmap('tab10')(0)  # First color for clients
    #agg_color = plt.get_cmap('tab10')(2)  # Third color for aggregated model
    fed_color = plt.get_cmap('tab10')(3)  # Fourth color for federated model
    #fed_roc_auc_color = plt.get_cmap('tab10')(4)  # Fifth color for federated model ROC AUC

    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D']  # Circle, square, triangle up, diamond
    metric_names = [ 'F1 score','Precision', 'Recall']
    fig = plt.figure(figsize=(8, 3))
    
    client_means, client_stds = calculate_mean_std(metrics,  'f1_score')
    plot_with_intervals(rounds, client_means, client_stds, 'Client Models F1 scores', client_color, 'f1_score', alpha=1, marker=markers[0])

    for idx, key in enumerate(metric_keys):
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        alpha = 1.0 if idx == 0 else 0.5 
        fed_values = [m['federated_model']['binary'][key] for m in metrics]
        plot_model_metric(rounds, fed_values, f'Federated Model {metric_names[idx]}', fed_color, linestyle, alpha=alpha, marker=marker)


    # Plot ROC AUC for the federated model
    fed_roc_auc_values = [m['federated_model']['roc_auc'] for m in metrics]
    plot_model_metric(rounds, fed_roc_auc_values, 'Federated Model ROC AUC', fed_color, line_styles[-1], alpha=0.5, marker=markers[-1])

    plt.xticks(np.arange(min(rounds), max(rounds)+1, 1.0))  # Set x-axis ticks to integers
    plt.xlabel('Communication Round')
    plt.ylabel('Metric Value')
    if legend:
        plt.legend()
    plt.show()

    return fig


def plot_clusters_with_kappa_pairs(model, kappa, save=False):

    plt.figure(figsize=(4, 4))

    # Use the join threshold from the parent
    join_threshold = model.parent.kappa_join
    num_clusters = model.parent.c

    colormap = 'tab10'
    cmap = cm.get_cmap(colormap)
    red_color = cmap(0) 

    for i in range(num_clusters):
        plot_cluster_merging(model, i, label=f'Cluster {i}', color=red_color, alpha=1)

    for i in range(num_clusters):

        row_kappa = kappa[i].clone()

        row_kappa[i] = 0  # Set model-connections to zero
        significant_indices = (row_kappa >= join_threshold).nonzero(as_tuple=False).flatten()

        if significant_indices.numel() == 0:
            continue

        # Iterate over all significant connections
        for idx in significant_indices:
            max_kappa_index = idx.item()

            cluster_i = model.parent.mu[i].cpu().numpy()[:2]
            cluster_j = model.parent.mu[max_kappa_index].cpu().numpy()[:2]
            plt.plot(
                [cluster_i[0], cluster_j[0]],
                [cluster_i[1], cluster_j[1]],
                linestyle='-',  # Dashed lines for all connections above the threshold
                color="red",
                linewidth=1.5
            )

    if save:
        output_directory = ".Images/Merging/Kappa"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        next_figure_number = model.get_next_figure_number(output_directory)
        output_filename = f"{output_directory}/kappa_connections_{next_figure_number}.pdf"

        plt.savefig(output_filename, bbox_inches='tight')

    plt.show()

def plot_cluster_merging(model, index, label, color, alpha=1):

    mu = model.parent.mu[index].cpu().detach().numpy()
    S = model.parent.S[index].cpu().detach().numpy() / model.parent.n[index].cpu().detach().numpy()

    mu_2d = mu[:2]
    S_2d = S[:2, :2]

    # Use the provided color for the cluster center
    plt.scatter(mu_2d[0], mu_2d[1], s=100, marker=MarkerStyle('.'), color=color, label=label, alpha=alpha)

    # Calculate ellipse parameters
    vals, vecs = np.linalg.eigh(S_2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    t = np.linspace(0, 2 * np.pi, 100)
    ell_radius_x = np.sqrt(vals[0]) * model.parent.num_sigma
    ell_radius_y = np.sqrt(vals[1]) * model.parent.num_sigma
    x = ell_radius_x * np.cos(t)
    y = ell_radius_y * np.sin(t)

    data = np.array([x, y])
    rotation_matrix = np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
        [np.sin(np.radians(theta)),  np.cos(np.radians(theta))]
    ])
    rotated_data = np.dot(rotation_matrix, data)
    translated_data = rotated_data + mu_2d[:, None]

    # Use the provided color for the ellipse
    plt.plot(translated_data[0], translated_data[1], color=color, lw=2, alpha=alpha)

def plot_cluster_data(metrics, rounds, legend= True):

    colormap = plt.get_cmap('tab10')
    color_clients = colormap(0)
    color_aggregated = colormap(2)
    color_federated = colormap(3)

    federated_clusters = [m['federated_model']['clusters'] for m in metrics]
    client_clusters = np.array([[cm['clusters'].cpu() for cm in m['client_metrics']] for m in metrics])
    client_clusters_mean = np.mean(client_clusters, axis=1)
    client_clusters_std = np.std(client_clusters, axis=1)

    fig = plt.figure(figsize=(8, 3))
    plot_with_intervals(rounds, client_clusters_mean, client_clusters_std, 'Average Client Clusters', color_clients, 'Number of Clusters', marker='s')
    #plt.plot(rounds, aggregated_clusters, marker='^', linestyle='-', color=color_aggregated, label='Aggregated Clusters')
    plt.plot(rounds, federated_clusters, marker='D', linestyle='-', color=color_federated, label='Federated Clusters')

    plt.xticks(np.arange(min(rounds), max(rounds)+1, 1.0))  # Set x-axis ticks to integers
    plt.xlabel('Communication Round')
    plt.ylabel('Number of Clusters')
    if legend:
        plt.legend()
    plt.show()

    return fig

def plot_all_features_upper_triangle(data, labels, model, N_max, num_sigma, colormap='tab10'):

    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    n_features = data.shape[1]

    unique_labels = np.unique(labels)
    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, len(unique_labels)))
    label_color_dict = dict(zip(unique_labels, label_colors))

    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            ax = axes[i, j]

            # Scatter plot of feature i vs feature j
            data_colors = [label_color_dict[label.item()] for label in labels]
            ax.scatter(data[:, j], data[:, i], c=data_colors, alpha=0.5)

            # Plotting ellipses for clusters
            for cluster_idx in range(model.c):
                if model.n[cluster_idx] > N_max:
                    mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                    S = model.S[cluster_idx].cpu().detach().numpy()
                    cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                    cov_submatrix = cov_matrix[[j, i]][:, [j, i]]
                    mu_subvector = mu_val[[j, i]]

                    vals, vecs = np.linalg.eigh(cov_submatrix)
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    factor = num_sigma
                    width, height = factor * np.sqrt(vals)
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none')
                    ax.add_patch(ell)

            ax.set_xlabel(f'Feature {j}')
            ax.set_ylabel(f'Feature {i}')
            ax.grid(True)
            axes[j, i].axis('off')
            if i == j:
                ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_all_features_combinations(data, labels, model, N_max, num_sigma, colormap='tab10'):
 
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    n_features = data.shape[1]
    unique_labels = np.unique(labels)
    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, len(unique_labels)))
    label_color_dict = dict(zip(unique_labels, label_colors))

    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: plot distribution of feature i
                ax.hist(data[:, i], bins=30, color='gray', alpha=0.7)
            else:
                # Scatter plot of feature i vs feature j
                data_colors = [label_color_dict[label.item()] for label in labels]
                ax.scatter(data[:, j], data[:, i], c=data_colors, alpha=0.5)

                # Plotting ellipses for clusters
                for cluster_idx in range(model.c):
                    if model.n[cluster_idx] > N_max:
                        mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                        S = model.S[cluster_idx].cpu().detach().numpy()
                        cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                        cov_submatrix = cov_matrix[[j, i]][:, [j, i]]
                        mu_subvector = mu_val[[j, i]]

                        vals, vecs = np.linalg.eigh(cov_submatrix)
                        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                        factor = num_sigma
                        width, height = factor * np.sqrt(vals)
                        ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none')
                        ax.add_patch(ell)

            ax.set_xlabel(f'Feature {j}')
            ax.set_ylabel(f'Feature {i}')
            ax.grid(True)

    plt.tight_layout()
    plt.show()

def fisher_score(data, labels, feature_idx1, feature_idx2):
    unique_labels = np.unique(labels)

    # Ensure feature indices are within the range of data's columns
    if feature_idx1 >= data.shape[1] or feature_idx2 >= data.shape[1]:
        raise ValueError("Feature indices are out of bounds.")

    means = []
    label_data_list = []  # List to store label data for each label
    for label in unique_labels:
        label_data = data[labels == label]  # Filter rows where label matches
        if label_data.size == 0:
            continue  # Skip if no data for this label
        means.append(np.mean(label_data[:, [feature_idx1, feature_idx2]], axis=0))
        label_data_list.append(label_data)  # Append label data to the list

    overall_mean = np.mean(data[:, [feature_idx1, feature_idx2]], axis=0)

    # Fisher Score calculation - ensure scalar values for s_b and s_w
    s_b = sum(len(ld) * np.sum((m - overall_mean)**2) for ld, m in zip(label_data_list, means))
    s_w = sum(np.sum((ld[:, [feature_idx1, feature_idx2]] - m)**2) for ld, m in zip(label_data_list, means))
    
    return s_b / s_w if s_w > 0 else 0


def select_unique_combinations(feature_combinations, fisher_scores, N_combinations):
    
    
    """Select top combinations with priority to unique features."""
    sorted_combinations = sorted(feature_combinations, key=lambda x: fisher_scores[x], reverse=True)
    selected_combinations = []
    used_features = set()

    for comb in sorted_combinations:
        if len(selected_combinations) >= N_combinations:
            break
        if comb[0] not in used_features and comb[1] not in used_features:
            selected_combinations.append(comb)
            used_features.update(comb)

    for comb in sorted_combinations:
        if len(selected_combinations) >= N_combinations:
            break
        if comb not in selected_combinations:
            selected_combinations.append(comb)
    # Sort the selected combinations by the lowest feature index
        ordered_combinations = sorted(selected_combinations, key=lambda x: (min(x), max(x)))

    return ordered_combinations


def plot_interesting_features(dataset, model, N_max, num_sigma, N_combinations=5, colormap='tab10', flag_unlabeled = True):
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels


    if len(data.shape) == 1:
        data = data.unsqueeze(0)

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)
    #clear_output(wait=True)
            
    #S = model.S[:model.c].clone().cpu().detach().numpy() / model.n[:model.c].cpu().detach().numpy()[:, np.newaxis, np.newaxis]
    #det_matrix = torch.exp(torch.linalg.slogdet(torch.tensor(S))[1])  # [1] is the log determinant
    #V_i = det_matrix ** (1/model.feature_dim)
    n_i = model.n[:model.c].clone().cpu().detach()
    # Assuming V_i is a NumPy array, convert it to a PyTorch tensor
    
    # Sort clusters for each class based on volume and select top 10
    top_clusters_per_class = {}
    for class_idx in range(model.num_classes):
        cluster_indices = torch.where(model.cluster_labels[:model.c, class_idx].cpu().detach())[0]

        # Use V_i directly if it is already a tensor
        sorted_indices = torch.argsort(n_i[cluster_indices], descending=False)[:100]
        top_clusters_per_class[class_idx] = cluster_indices[sorted_indices]

    n_features = data.shape[1]
    unique_labels = model.num_classes

    
    # Generate colors using the colormap for the reduced number of labels
    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels-1))

    # Convert to a list for easy manipulation
    label_colors = label_colors.tolist()

    # Insert gray color at the beginning of the color list
    #unlabelled_color = (0.5, 0.5, 0.5, 1)  # Gray color in RGBA
    unlabelled_color = (1, 0, 0, 1)  # Red color in RGBA
    label_colors.insert(0, unlabelled_color)
    
    label_color_dict = dict(zip(range(unique_labels), label_colors))
    try:
            data_colors = [label_color_dict[label.item()-1] for label in labels]
    except:
        print("Error label color")
        
    # Calculate the maximum number of feature combinations
    n_features = data.shape[1]
    max_feature_combinations = comb(n_features, 2)

    # Adjust N_combinations if the maximum possible combinations are fewer
    if max_feature_combinations < N_combinations:
        N_combinations = max_feature_combinations
            
    # Calculate Fisher Scores for all combinations, including cluster centers
    feature_combinations = list(combinations(range(n_features), 2))
    fisher_scores = {}
    
    mu = model.mu[:model.c].cpu().detach().numpy()  # Adjust this based on how you access the cluster centers
    for combination in feature_combinations:
        # Select the data for the current feature combination
        data_comb = data[:, combination]

        # Extract the corresponding cluster centers for the current feature combination
        mu_comb = mu[:, combination]

        # Determine the labels for each cluster center
        cluster_center_labels = torch.where(model.cluster_labels[:model.c])[1].cpu().numpy()

        # Ensure cluster_center_labels has the same length as mu_comb (number of clusters)
        assert len(cluster_center_labels) == mu_comb.shape[0], "Number of cluster centers must match number of labels."

        # Combine the data with the cluster centers
        augmented_data = np.vstack([data_comb, mu_comb])

        # Combine original labels with cluster center labels
        augmented_labels = np.concatenate([labels, cluster_center_labels])

        # Check if the number of augmented data points matches the number of augmented labels
        assert augmented_data.shape[0] == len(augmented_labels), "The number of data points must match the number of labels."

        # Calculate the Fisher score for the augmented data
        fisher_scores[combination] = fisher_score(augmented_data, augmented_labels, 0, 1)


    # Select top N combinations with priority to unique features
    top_combinations = select_unique_combinations(feature_combinations, fisher_scores, N_combinations)

    # Plotting logic
    rows = int(np.ceil(np.sqrt(N_combinations)))
    cols = rows if rows * (rows - 1) < N_combinations else rows - 1

    # Set up the subplot grid – all plots in a single row
    fig, axes = plt.subplots(1, N_combinations,  figsize=(4*N_combinations, 4), constrained_layout=True)  # Adjust figure size as needed

    # Flatten the axes array for easy indexing (if N_combinations is 1, wrap axes in a list)
    if N_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (feature_idx1, feature_idx2) in enumerate(top_combinations):
        ax = axes[idx]
        sample_size = max(0.1, min(10, 1000/np.unique(data[:, feature_idx1]).shape[0]))
        ax.scatter(data[:, feature_idx1], data[:, feature_idx2], c=data_colors, alpha=0.7, s=sample_size)

        for class_idx, top_clusters in top_clusters_per_class.items():
            for cluster_idx in top_clusters:
                
                mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                mu_subvector = mu_val[[feature_idx1, feature_idx2]]
                
                if model.n[cluster_idx] > N_max:

                    # Get the mean and covariance of the cluster for the current feature pair
                    S = model.S[cluster_idx].cpu().detach().numpy()
                    cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                    cov_submatrix = cov_matrix[[feature_idx1, feature_idx2]][:, [feature_idx1, feature_idx2]]
   
                    # Eigen decomposition for the ellipse orientation
                    vals, vecs = np.linalg.eigh(cov_submatrix)
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                    # Determine the width and height of the ellipse based on eigenvalues
                    factor = num_sigma  # Number of standard deviations to plot
                    width, height = factor * np.sqrt(vals)

                    # Determine the color for the ellipse
                    if model.cluster_labels[cluster_idx, 0].item() == 1:
                        ellipse_color = label_color_dict[0]
                    else:
                        ellipse_color = label_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]

                    ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color)
                    dark_factor = 0.8  # Factor to darken the color
                    darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, 
                                            ellipse_color_rgba[1] * dark_factor, 
                                            ellipse_color_rgba[2] * dark_factor, 1)

                    # Create and add the ellipse patch
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), 
                                    width=width, height=height, 
                                    angle=angle, edgecolor=darker_ellipse_color, 
                                    lw=2, facecolor='none')
                    ax.add_patch(ell)

                # Mark the cluster center
                ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=10, marker='.')

        # Set scientific notation for axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        ax.set_xlabel(f"X {feature_idx1 + 1}")
        ax.set_ylabel(f"X {feature_idx2 + 1}")
        ax.grid(False)

    #plt.tight_layout()
    #plt.show()
    
    return fig

def plot_first_feature_combinations(dataset, model, N_max, num_sigma, N_combinations=5, colormap='tab10', flag_unlabeled = False, flag_axis = True, title = None):
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    
    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)

    # Randomly sample up to 10,000 indices
    num_samples = min(10000, len(data))
    sampled_indices = np.random.choice(len(data), num_samples, replace=False)
    
    # Subset data and labels using the sampled indices
    data = data[sampled_indices]
    labels = labels[sampled_indices]
    
    
    n_i = model.n[:model.c].clone().cpu().detach()
    
    top_clusters_per_class = {}
    for class_idx in range(model.num_classes):
        cluster_indices = torch.where(model.cluster_labels[:model.c, class_idx].cpu().detach())[0]
        sorted_indices = torch.argsort(n_i[cluster_indices], descending=True)[:1000]
        top_clusters_per_class[class_idx] = cluster_indices[sorted_indices]

    n_features = data.shape[1]


    unique_labels = len(np.unique(labels))+1 #model.num_classes

    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels-1))
    label_colors = label_colors.tolist()
    unlabelled_color = [1, 0, 0, 1] 
    label_colors.insert(0, unlabelled_color)
    label_color_dict = dict(zip(range(unique_labels), label_colors))
    
    # Handle color assignment for data points
    if flag_unlabeled:
        #data_colors = [blue_color] * len(labels)
        data_colors = [label_color_dict[label.item()+1] for label in labels]
    else:
        try:
            data_colors = [label_color_dict[label.item()] for label in labels]
        except Exception as e:
            print(f"Error in label color assignment: {e}")

    # Directly create combinations with the first feature and the subsequent features
    top_combinations = [(0, i) for i in range(1, n_features)]
    actual_combinations = min(len(top_combinations), N_combinations)
    
    # Set rows to 1 and cols to actual_combinations for a single row of plots
    rows = 1
    cols = actual_combinations

    # Create a single row of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), constrained_layout=True)

    if actual_combinations == 1:
        axes = [axes]
    elif actual_combinations > 1:
        axes = axes.flatten()

    for idx, (feature_idx1, feature_idx2) in enumerate(top_combinations[:actual_combinations]):
        ax = axes[idx]
        sample_size = max(1, min(10, 1000/np.unique(data[:, feature_idx1]).shape[0]))
        ax.scatter(data[:, feature_idx1], data[:, feature_idx2], c=data_colors, alpha=0.7, s=sample_size)

        for class_idx, top_clusters in top_clusters_per_class.items():
            for cluster_idx in top_clusters:
                
                mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                mu_subvector = mu_val[[feature_idx1, feature_idx2]]
                
                if model.n[cluster_idx] > N_max:

                    # Get the mean and covariance of the cluster for the current feature pair
                    S = model.S[cluster_idx].cpu().detach().numpy()
                    cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                    cov_submatrix = cov_matrix[[feature_idx1, feature_idx2]][:, [feature_idx1, feature_idx2]]
   
                    # Eigen decomposition for the ellipse orientation
                    vals, vecs = np.linalg.eigh(cov_submatrix)
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                    # Determine the width and height of the ellipse based on eigenvalues
                    factor = num_sigma  # Number of standard deviations to plot
                    width, height = factor * np.sqrt(vals)

                    # Determine the color for the ellipse
                    if model.cluster_labels[cluster_idx, 0].item() == 1:
                        ellipse_color = label_color_dict[0]
                    else:
                        ellipse_color = label_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]

                    ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color)
                    dark_factor = 0.8  # Factor to darken the color
                    darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, 
                                            ellipse_color_rgba[1] * dark_factor, 
                                            ellipse_color_rgba[2] * dark_factor, 1)

                    # Create and add the ellipse patch
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), 
                                    width=width, height=height, 
                                    angle=angle, edgecolor=darker_ellipse_color, 
                                    lw=2, facecolor='none')
                    ax.add_patch(ell)

                # Mark the cluster center
                ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=50, marker='.')

        # Set scientific notation for axes
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
        ax.set_xlabel(rf"$x_{feature_idx1 + 1}$")
        ax.set_ylabel(rf"$x_{feature_idx2 + 1}$")
        ax.grid(False)

        if not flag_axis:
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_xlabel('')  # Remove x-axis label

    # Remove any extra empty subplots
    for idx in range(actual_combinations, len(axes)):
        fig.delaxes(axes[idx])

    # Add a figure title if passed
    if title:
        fig.suptitle(title)

    return fig

def save_figure(figure, filename, format='pdf'):
    figure.savefig(filename, format=format, bbox_inches='tight')
    print(f"Figure saved as {filename} in {format} format.")
    
def plot_client_data_distribution(y_true_split, num_clients, num_classes):
    """
    Plots the class distribution for each client using the true labels stored in `y_true_split`.
    """

    # Initialize a matrix to store class distribution per client
    client_class_distribution = np.zeros((num_clients, num_classes))

    # Calculate the class distribution for each client
    for client_idx in range(num_clients):
        client_labels = y_true_split[client_idx].numpy()  # Get labels for the client
        class_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_distribution[client_idx, :] = class_counts

    # Plotting the class distribution for each client
    plt.figure(figsize=(6, 4))
    for i in range(num_clients):
        plt.bar(np.arange(num_classes) + i * 0.1, client_class_distribution[i], width=0.1, label=f'Client {i+1}')

    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.title('Non-IID Class Distribution Across Clients')
    plt.legend()
    plt.show()


def plot_clients_and_federated_model(dataset_list, model_list, federated_model, N_max, round_num, colormap='tab10', flag_axis=False, flag_legend=True):
    # Create main figure
    fig, ax = plt.subplots(figsize=(4, 4))

    cmap = plt.get_cmap(colormap)
    if len(dataset_list) == 3:
        colors = [cmap(0), cmap(2), cmap(3)]  # Blue, Green, Red in Tab10
    else:
        colors = cmap(np.linspace(0, 1, len(dataset_list)))

    # Initialize min/max variables for axis limits
    min_x, max_x, min_y, max_y = 0, 1, 0, 1

    legend_elements = []

    # Plot each client's data and clusters
    for client_idx, (dataset, model) in enumerate(zip(dataset_list, model_list)):
        data, labels = dataset
        data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data

        min_x, max_x = min(min_x, data[:, 0].min()), max(max_x, data[:, 0].max())
        min_y, max_y = min(min_y, data[:, 1].min()), max(max_y, data[:, 1].max())
        
        scatter = ax.scatter(data[:, 0], data[:, 1],
                     color=colors[client_idx],  # ← use color, not c
                     s=10, alpha=0.5)
        legend_elements.append((scatter, f"Client {client_idx + 1} Data"))

        # Plot clusters for the client model
        n_i = model.n[:model.c].cpu().detach()
        for cluster_idx in range(model.c):
            if model.n[cluster_idx] > N_max:
                mu_val = model.mu[cluster_idx].cpu().detach().numpy()[:2]
                S = model.S[cluster_idx].cpu().detach().numpy()
                cov_matrix = S / model.n[cluster_idx].cpu().detach().numpy()
                cov_submatrix = cov_matrix[:2, :2]

                vals, vecs = np.linalg.eigh(cov_submatrix)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals)

                ell = Ellipse(xy=mu_val, width=width, height=height, angle=angle, edgecolor=colors[client_idx], lw=2, 
                              facecolor='none', linestyle='-', alpha=1)
                ax.add_patch(ell)
                if cluster_idx == 0:  # Add to legend only once per client
                    legend_elements.append((ell, f"Client {client_idx + 1} Clusters"))

                min_x, max_x = min(min_x, mu_val[0] - width / 2), max(max_x, mu_val[0] + width / 2)
                min_y, max_y = min(min_y, mu_val[1] - height / 2), max(max_y, mu_val[1] + height / 2)

    # Plot the federated model's clusters
    federated_n_i = federated_model.n[:federated_model.c].cpu().detach()
    for cluster_idx in range(federated_model.c):
        if federated_n_i[cluster_idx] > N_max:
            mu_val = federated_model.mu[cluster_idx].cpu().detach().numpy()[:2]
            S = federated_model.S[cluster_idx].cpu().detach().numpy()
            cov_matrix = S / federated_n_i[cluster_idx].cpu().detach().numpy()
            cov_submatrix = cov_matrix[:2, :2]

            vals, vecs = np.linalg.eigh(cov_submatrix)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)

            ell = Ellipse(xy=mu_val, width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none')
            ax.add_patch(ell)
            if cluster_idx == 0:  # Add to legend only once
                legend_elements.append((ell, "Federated Clusters"))

            center = ax.scatter(mu_val[0], mu_val[1], color='black', s=100, marker='.')
            if cluster_idx == 0:  # Add to legend only once
                legend_elements.append((center, "Federated Centers"))

            min_x, max_x = min(min_x, mu_val[0] - width / 2), max(max_x, mu_val[0] + width / 2)
            min_y, max_y = min(min_y, mu_val[1] - height / 2), max(max_y, mu_val[1] + height / 2)

    # Set axis limits and labels
    y_range = max(1, max_y) - min(0, min_y)
    ax.set_xlim(min(0, min_x), max(1, max_x))
    ax.set_ylim(min(0, min_y - 0.05 * y_range), max(1, max_y))

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(False)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    if not flag_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')


    ax.text(0.5, 0.005, f"Round {round_num+1}", weight='bold', horizontalalignment='center', verticalalignment='bottom', 
        transform=ax.transAxes, fontsize=14)

    plt.tight_layout()

    if flag_legend:
        # Create figure without margins
        legend_fig = plt.figure(figsize=(12, 0.8))  # Start with a larger figure
        legend_ax = legend_fig.add_axes([0, 0, 1, 1])
        legend_ax.axis('off')

        # Create bold font
        bold_font = fm.FontProperties(weight='bold', size=14)
        
        # Create horizontal legend with minimal spacing
        legend = legend_ax.legend(*zip(*legend_elements), loc='center', ncol=len(legend_elements),
                                  frameon=False, prop=bold_font, 
                                  handlelength=1.2, handleheight=1.2,
                                  handletextpad=0.4, columnspacing=1.0,
                                  borderpad=0, borderaxespad=0)

        # Adjust legend item spacing
        legend._loc = 3  # lower left
        for text in legend.get_texts():
            text.set_fontweight('bold')
        legend._ncol = len(legend_elements)
        
        # Define padding value here
        padding = 2  # pixels

        # Tighten the legend box
        legend_fig.canvas.draw()  # Ensure the figure is drawn
        bbox = legend.get_window_extent()
        tight_bbox = Bbox([[bbox.x0 - padding, bbox.y0 - padding], [bbox.x1 + padding, bbox.y1 + padding]])
        
        # Set the figure size to match the tight bounding box
        legend_fig.set_size_inches(tight_bbox.width / legend_fig.dpi, tight_bbox.height / legend_fig.dpi)

    else:
        legend_fig = None

    return fig, legend_fig

import os
from matplotlib.colors import ListedColormap

def plot_results_metrics(X_test, y_test, centers, assignments, name, execution_time, silhouette, ari, nmi):

    feature_dim = X_test.shape[1]

    directory = ".Images/related_methods/noniid"
    os.makedirs(directory, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    custom_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#ff7f0e',  # orange
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]

    custom_cmap = ListedColormap(custom_colors)
    num_clusters = len(np.unique(assignments))
    for cluster in np.unique(assignments):
        cluster_points = X_test[assignments == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=custom_colors[cluster % len(custom_colors)], alpha=1/np.sqrt(feature_dim))
    ax.scatter(centers[:, 0], centers[:, 1], s=50, color='black', marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
    ax.set_ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
    plt.axis('tight')

    if 0:
        # Add execution time below the metrics
        ax.text(0.1, 0.001, f't:{execution_time:.2f}s', 
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, 
                color='black', fontsize=12, fontweight='bold')

        ax.text(0.999, 0.101, f'S:{silhouette:.3f}', 
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, 
                color='black', fontsize=12, fontweight='bold')

        ax.text(0.999, 0.051, f'ARI:{ari:.3f}', 
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, 
                color='black', fontsize=12, fontweight='bold')

        ax.text(0.999, 0.001, f'NMI:{nmi:.3f}', 
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, 
                color='black', fontsize=12, fontweight='bold')

    metrics_text = f'↑ S:{silhouette:.2f} | ARI:{ari:.2f} | NMI:{nmi:.2f}'

    # Display the metrics text at the bottom center of the plot
    ax.text(0.5, 0.001, metrics_text,
            verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes, 
            color='black', fontsize=11, fontweight='bold')
    time_text = f'↓ Time:{execution_time:.2f}s '
    ax.text(0.999, 0.920, time_text,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, 
            color='black', fontsize=12, fontweight='bold')

    pdf_path = os.path.join(directory, f'{name}.pdf')
    png_path = os.path.join(directory, f'{name}.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

from collections import Counter


def plot_results_correct(X_test, y_test, centers, assignments, name, execution_time, covariance=None, method='pca'):
    # Ensure the directory exists
    directory = ".Images/related_methods/scalability"
    os.makedirs(directory, exist_ok=True)
    
    # Randomly sample 10,000 indices
    num_samples = min(1000, len(X_test))
    sampled_indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Subset data using sampled indices
    X_test_sampled = X_test[sampled_indices]
    y_test_sampled = y_test[sampled_indices]
    assignments_sampled = assignments[sampled_indices]
    
    # Apply Seaborn dark palette style
    fig, ax = plt.subplots(figsize=(4, 4))  # Set figure size to be square

    # Set the background color of the plot to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    # Use the first two features directly if they are available
    X_test_reduced = X_test_sampled[:, :2]
    centers_reduced = centers[:, :2]
    
    correct_points = []
    incorrect_points = []

    for cluster in np.unique(assignments_sampled):
        cluster_points = X_test_reduced[assignments_sampled == cluster]
        cluster_labels = y_test_sampled[assignments_sampled == cluster]
        most_common_label = Counter(cluster_labels).most_common(1)[0][0]
        correct_points.extend(cluster_points[cluster_labels == most_common_label])
        incorrect_points.extend(cluster_points[cluster_labels != most_common_label])


    correct_points = np.array(correct_points)
    incorrect_points = np.array(incorrect_points)

    # Plot correct points in green
    if correct_points.size > 0:
        ax.scatter(correct_points[:, 0], correct_points[:, 1], s=5, color='green', alpha=0.75)
    
    # Plot incorrect points in red
    if incorrect_points.size > 0:
        ax.scatter(incorrect_points[:, 0], incorrect_points[:, 1], s=5, color='red', alpha=0.75)
    
    # Plot cluster centers
    ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1], s=50, color='black', marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(X_test_reduced[:, 0].min() - 1, X_test_reduced[:, 0].max() + 1)
    ax.set_ylim(X_test_reduced[:, 1].min() - 1, X_test_reduced[:, 1].max() + 1)
    plt.axis('tight')
    ax.text(0.999, 0.001, f'{execution_time:.2f}s', 
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, 
            color='black', fontsize=14, fontweight='bold')
    pdf_path = os.path.join(directory, f'{name}.pdf')
    png_path = os.path.join(directory, f'{name}.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_results_acc(X_test, y_test, centers, assignments, name, execution_time, covariance=None, method='pca'):

    directory = ".Images/related_methods/scalability"
    os.makedirs(directory, exist_ok=True)
    num_samples = min(1000, len(X_test))
    sampled_indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_test_sampled = X_test[sampled_indices]
    assignments_sampled = assignments[sampled_indices]
    fig, ax = plt.subplots(figsize=(4, 4))  # Set figure size to be square

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    custom_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#ff7f0e',  # orange
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
    ]

    for cluster in np.unique(assignments_sampled):
        cluster_points = X_test_sampled[assignments_sampled == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=custom_colors[cluster % len(custom_colors)], alpha=0.75)
    
    ax.scatter(centers[:, 0], centers[:, 1], s=50, color='black', marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(X_test_sampled[:, 0].min() - 1, X_test_sampled[:, 0].max() + 1)
    ax.set_ylim(X_test_sampled[:, 1].min() - 1, X_test_sampled[:, 1].max() + 1)


    plt.axis('tight')
    ax.text(0.999, 0.001, f'{execution_time:.2f}s', 
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, 
            color='black', fontsize=14, fontweight='bold')

    plt.show()


from matplotlib.patches import Ellipse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def visualize_clients_data(train_loaders_by_round, num_clients, federated_model, N_max, metrics, colormap='tab10', flag_axis=False): 

    fig, ax = plt.subplots(figsize=(4, 4))  # Keep the figure size exactly the same
    
    # Get the colormap for clients
    cmap = plt.get_cmap(colormap)
    
    # Adjust color selection from Tab10 for red, green, and blue if num_clients == 3
    if num_clients == 3:
        colors = [cmap(0), cmap(2), cmap(3)]  # Blue, Green, Red in Tab10
    else:
        # Generate a color for each client, spread evenly
        colors = cmap(np.linspace(0, 1, num_clients))

    # Initialize min/max variables for axis limits
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

    # Plot the data points for each client (aggregating all rounds)
    for client_idx in range(num_clients):
        client_color = colors[client_idx]  # Unique color for each client

        # Initialize lists to gather data for all rounds of this client
        X_client_np = []
        
        # Iterate through all rounds and collect the data
        for round_idx in range(len(train_loaders_by_round[client_idx])):
            round_loader = train_loaders_by_round[client_idx][round_idx]

            # Iterate through all batches in the round's DataLoader
            for X_batch, y_batch in round_loader:
                # Convert to numpy for plotting
                X_batch_np = X_batch.cpu().detach().numpy() if isinstance(X_batch, torch.Tensor) else X_batch
                X_client_np.append(X_batch_np)

        # Concatenate all rounds' data for this client
        X_client_np = np.concatenate(X_client_np, axis=0)

        # Plot data for this client with the assigned color
        ax.scatter(X_client_np[:, 0], X_client_np[:, 1], color=client_color, s=10, label=f"Client {client_idx + 1} Data", alpha=0.9)

        # Update axis limits
        min_x = min(min_x, X_client_np[:, 0].min())
        max_x = max(max_x, X_client_np[:, 0].max())
        min_y = min(min_y, X_client_np[:, 1].min())
        max_y = max(max_y, X_client_np[:, 1].max())

    # Plot the federated model's clusters with sigma = 3 and solid line
    federated_n_i = federated_model.n[:federated_model.c].cpu().detach()
    for cluster_idx in range(federated_model.c):
        if federated_n_i[cluster_idx] > N_max:
            mu_val = federated_model.mu[cluster_idx].cpu().detach().numpy()[:2]
            S = federated_model.S[cluster_idx].cpu().detach().numpy()
            cov_matrix = S / federated_n_i[cluster_idx].cpu().detach().numpy()
            cov_submatrix = cov_matrix[:2, :2]

            # Eigen decomposition for ellipse
            vals, vecs = np.linalg.eigh(cov_submatrix)
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)  # Use sigma = 3 for federated model

            # Plot solid ellipse for the federated model
            ell = Ellipse(xy=mu_val, width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none', 
                          label="Federated Clusters")
            ax.add_patch(ell)

            # Mark the cluster center
            ax.scatter(mu_val[0], mu_val[1], color='black', s=100, marker='.', label="Federated Centers")

            # Update axis limits based on federated model
            min_x = min(min_x, mu_val[0] - width / 2)
            max_x = max(max_x, mu_val[0] + width / 2)
            min_y = min(min_y, mu_val[1] - height / 2)
            max_y = max(max_y, mu_val[1] + height / 2)

    ax.grid(False)
    
    if not flag_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    # Display metrics at the bottom within the extended portion of the y-axis
    metric_text = f"ARI:{metrics['ARI']:.2f} | NMI:{metrics['NMI']:.2f} | ACC:{metrics['ACC']*100:.1f}%"
    ax.text(0.5, -0.05,  metric_text, weight='bold', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    plt.tight_layout()

    return fig

def plot_federated_streaming(round_results, name, save_dir=".Images/Streaming/"):

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract data
    rounds = [result["round"] for result in round_results]
    num_clusters = [result["num_clusters"] for result in round_results]
    ari = [result["metrics"].get("ARI", None) for result in round_results]
    nmi = [result["metrics"].get("NMI", None) for result in round_results]
    acc = [result["metrics"].get("ACC", None) for result in round_results]  # Accuracy already between 0 and 1

    # Plot metrics
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, ari, marker='o', label="ARI", color="orange")
    plt.plot(rounds, nmi, marker='s', label="NMI", color="green")
    plt.plot(rounds, acc, marker='^', label="ACC ×100%", color="red")  # ACC plotted between 0 and 1
    plt.xlabel("Rounds")
    plt.ylabel("Metric Values")
    plt.xticks(rounds)  # Ensure x-axis is treated as discrete
    plt.margins(0)  # Tight x-axis
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    metrics_path_png = os.path.join(save_dir, f"{name}_metrics_plot.png")
    metrics_path_pdf = os.path.join(save_dir, f"{name}_metrics_plot.pdf")
    plt.savefig(metrics_path_png)
    plt.savefig(metrics_path_pdf)
    print(f"Saved metrics plot to {metrics_path_png} and {metrics_path_pdf}")
    plt.close()

    # Plot number of clusters
    plt.figure(figsize=(8, 2))
    plt.plot(rounds, num_clusters, marker='o', label="Number of Clusters", color="blue")
    plt.xlabel("Rounds")
    plt.ylabel("Clusters")
    plt.xticks(rounds)  
    plt.margins(0)  
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    clusters_path_png = os.path.join(save_dir, f"{name}_clusters_plot.png")
    clusters_path_pdf = os.path.join(save_dir, f"{name}_clusters_plot.pdf")
    plt.savefig(clusters_path_png)
    plt.savefig(clusters_path_pdf)
    print(f"Saved clusters plot to {clusters_path_png} and {clusters_path_pdf}")
    plt.close()
