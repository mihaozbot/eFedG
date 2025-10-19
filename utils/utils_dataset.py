import torch
import numpy as np
from sklearn.utils import shuffle
import random
from utils.utils_plots import plot_client_data_distribution
from torch.utils.data import DataLoader, TensorDataset, Subset



def non_iid_data(X, y, num_clients):

    # Move class 0 data from first client to second client
    if num_clients > 1:
        class_0_indices = y[0] == 0
        if any(class_0_indices):
            X[1] = np.concatenate((X[1], X[0][class_0_indices]))
            y[1] = np.concatenate((y[1], y[0][class_0_indices]))
            X[0] = X[0][~class_0_indices]
            y[0] = y[0][~class_0_indices]

    # Move class 1 data from second client to third client
    if num_clients > 2:
        class_1_indices = y[1] == 1
        if any(class_1_indices):
            X[2] = np.concatenate((X[2], X[1][class_1_indices]))
            y[2] = np.concatenate((y[2], y[1][class_1_indices]))
            X[1] = X[1][~class_1_indices]
            y[1] = y[1][~class_1_indices]

    return X, y

def display_dataset_split(client_data, test_dataset):

    _, y_test = test_dataset

    for i, (_, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client.numpy(), return_counts=True)
        print(f"Client {i + 1}: {dict(zip(unique, counts))}")

    unique_test, counts_test = np.unique(y_test.numpy(), return_counts=True)
    print(f"Test Set: {dict(zip(unique_test, counts_test))}")
    combined_y = np.concatenate([y_client.numpy() for _, y_client in client_data] + [y_test.numpy()])

    unique, counts = np.unique(combined_y, return_counts=True)
    combined_counts = dict(zip(unique, counts))

    print("\nCombined Number of Samples per Class:")
    for class_label, count in combined_counts.items():
        print(f"Class {class_label}: {count} samples")
    total_samples = sum(combined_counts.values())
    print(f"\nTotal Number of Samples Across All Datasets: {total_samples}")


def prepare_k_fold_non_iid_dataset(X, y, train_index, test_index, num_clients):

    train_index = shuffle(train_index)
    test_index = shuffle(test_index)

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    X_train_split = np.array_split(X_train, num_clients)
    y_train_split = np.array_split(y_train, num_clients)

    # Move class 0 data from first client to second client
    X_train_split, y_train_split = non_iid_data(X_train_split, y_train_split, num_clients)
    
    train_data = [(torch.tensor(X_train_split[i], dtype=torch.float32), 
                    torch.tensor(y_train_split[i], dtype=torch.int64)) 
                   for i in range(num_clients)]
    test_data = (torch.tensor(X_test, dtype=torch.float32), 
                 torch.tensor(y_test, dtype=torch.int64))
    all_data = (torch.tensor(X, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.int64))

    return train_data, test_data, all_data


def prepare_k_fold_federated_dataset_subset(X, y, train_index, test_index, num_clients, percentage_used):

    #train_index = shuffle(train_index)
    #test_index = shuffle(test_index)

    num_train_samples = int(percentage_used*num_clients*len(train_index))
    train_index_subset = train_index[:num_train_samples]
    X_train_subset, y_train_subset = X[train_index_subset], y[train_index_subset]
    X_test, y_test = X[test_index], y[test_index]

    X_train_split = np.array_split(X_train_subset, num_clients)
    y_train_split = np.array_split(y_train_subset, num_clients)

    train_data = [(torch.tensor(X_train_split[i], dtype=torch.float32), 
                   torch.tensor(y_train_split[i], dtype=torch.int64)) 
                  for i in range(num_clients)]
    test_data = (torch.tensor(X_test, dtype=torch.float32), 
                 torch.tensor(y_test, dtype=torch.int64))
    all_data = (torch.tensor(X, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.int64))

    return train_data, test_data, all_data


def prepare_k_fold_federated_dataset(X, y, train_index, test_index, num_clients, batch_size=32, unlabeled=0, use_unlabeled=True):


    X = X.clone().detach()
    y = y.clone().detach()

    X_train_subset, y_train_subset = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    y_train_subset += 1 # (unlabeled> 0)

    if unlabeled > 0:
        classes = torch.unique(y_train_subset)
        unlabeled_indices = []

        for cls in classes:
            indices_of_class = (y_train_subset == cls).nonzero(as_tuple=True)[0]
            if len(indices_of_class) == 0:
                print(f"No samples found for class {cls}.")
                continue
            num_unlabeled = int(len(indices_of_class) * unlabeled)
            num_unlabeled = min(num_unlabeled, len(indices_of_class))
            unlabeled_indices_cls = random.Random(42).sample(indices_of_class.tolist(), num_unlabeled)
            unlabeled_indices.extend(unlabeled_indices_cls)

        if use_unlabeled:
            y_train_subset[unlabeled_indices] = 0  # Mark as unlabeled
        else:
            mask = torch.ones(len(y_train_subset), dtype=torch.bool)
            mask[unlabeled_indices] = False
            labeled_indices = torch.arange(len(y_train_subset))[mask].tolist()
            X_train_subset = X_train_subset[labeled_indices]
            y_train_subset = y_train_subset[labeled_indices]

        # Check that each class has at least one labeled instance
        if unlabeled != 1:
            for cls in classes:
                labeled_instances = (y_train_subset == cls).nonzero(as_tuple=True)[0]
                if len(labeled_instances) == 0:
                    raise ValueError(f"Class {cls} has no labeled instances after unlabeling process.")
            

    X_train_split = torch.split(X_train_subset, int(len(X_train_subset) / num_clients))
    y_train_split = torch.split(y_train_subset, int(len(y_train_subset) / num_clients))


    train_dataset = TensorDataset(X_train_subset, y_train_subset)
    train_loaders = [DataLoader(TensorDataset(X_train_split[i], y_train_split[i]), batch_size=batch_size, shuffle=True, pin_memory=True) for i in range(num_clients)]
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)#, num_workers=4

    all_data = (X, y)
    train_data = (X_train_split, y_train_split)
    
    return train_loaders, train_dataset, test_loader, test_dataset, all_data, train_data

def prepare_iid_federated_dataset_for_clustering(X, y, num_clients, batch_size=32):
     
    X = X.clone().detach() 
    y = y.clone().detach()
    num_classes = len(torch.unique(y))
    y_train = torch.zeros_like(y)
    X_train_split = torch.split(X, int(len(X) / num_clients)) 
    y_train_split = torch.split(y_train, int(len(y_train) / num_clients)) 
    train_datasets = [TensorDataset(X_train_split[i], y_train_split[i]) for i in range(num_clients)] 
    train_loaders = [DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True) for i in range(num_clients)] 
    test_dataset = TensorDataset(X, y) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True) 
    all_data = (X, y) 
    train_data = (X_train_split, y_train_split) 
    
    plot_client_data_distribution(y_train_split, num_clients, num_classes)
    
    return train_loaders, train_datasets, test_loader, test_dataset, all_data, train_data

def prepare_non_iid_federated_dataset_for_clustering(X, y, num_clients, alpha=0.5, batch_size=32):

    y = y.clone().detach()

    X_train_split = [[] for _ in range(num_clients)]
    y_train_split = [[] for _ in range(num_clients)]
    y_true_split = [[] for _ in range(num_clients)]  # Store the true labels for plotting and testing

    num_classes = len(torch.unique(y))

    # Use Dirichlet distribution to generate skewed class distributions for each client
    class_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    for c in range(num_classes):
        indices = torch.where(y == c)[0]
        indices = indices[torch.randperm(len(indices))]  # Shuffle indices

        proportions = class_distribution[c]
        proportions = np.array(proportions * len(indices), dtype=int)
        proportions[-1] = len(indices) - np.sum(proportions[:-1])

        # Split indices based on the calculated proportions
        class_splits = torch.split(indices, proportions.tolist())
        for client_idx, client_data in enumerate(class_splits):
            X_train_split[client_idx].append(X[client_data])
            y_train_split[client_idx].append(torch.zeros_like(y[client_data]))  # Set to 0 for training
            y_true_split[client_idx].append(y[client_data])  # Keep the true labels for plotting/testing

    X_train_split = [torch.cat(client_data) for client_data in X_train_split]
    y_train_split = [torch.cat(client_labels) for client_labels in y_train_split]
    y_true_split = [torch.cat(client_labels) for client_labels in y_true_split]

    for i in range(num_clients):
        if X_train_split[i].shape[0] == 0:
            donor = max(range(num_clients), key=lambda j: X_train_split[j].shape[0])
            X_train_split[i] = X_train_split[donor][:1]
            y_train_split[i] = y_train_split[donor][:1]
            y_true_split[i]  = y_true_split[donor][:1]
            X_train_split[donor] = X_train_split[donor][1:]
            y_train_split[donor] = y_train_split[donor][1:]
            y_true_split[donor]  = y_true_split[donor][1:]
            
    train_datasets = [TensorDataset(X_train_split[i], y_train_split[i]) for i in range(num_clients)]
    train_loaders = [DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True) for i in range(num_clients)]

    X_test = torch.cat(X_train_split)
    y_test = torch.cat(y_true_split)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    plot_client_data_distribution(y_true_split, num_clients, num_classes)

    all_data = (X, y)
    train_data = (X_train_split, y_train_split)

    return train_loaders, train_datasets, test_loader, test_dataset, all_data, train_data

def prepare_non_iid_federated_streaming_dataset(X, y, num_clients, batch_size=32, num_rounds=1, alpha=0.5, flag_sort=True, swap_range=100):
    #Basically for streaming, I need to add some order to the data not just random

    y = y.clone().detach()
    X = X.clone().detach()

    num_classes = len(torch.unique(y))
    class_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    X_train_split = [[] for _ in range(num_clients)]
    y_train_split = [[] for _ in range(num_clients)]
    y_true_split = [[] for _ in range(num_clients)]  # Store the true labels for testing

    for class_idx in range(num_classes):
        class_indices = torch.where(y == class_idx)[0]
        class_proportions = (class_distribution[class_idx] * len(class_indices)).astype(int)
        class_proportions[-1] = len(class_indices) - class_proportions[:-1].sum()
        split_indices = torch.split(class_indices, class_proportions.tolist())

        for client_idx in range(num_clients):
            X_train_split[client_idx].append(X[split_indices[client_idx]])
            y_train_split[client_idx].append(y[split_indices[client_idx]])
            y_true_split[client_idx].append(y[split_indices[client_idx]])

    for client_idx in range(num_clients):
        X_train_split[client_idx] = torch.cat(X_train_split[client_idx])
        y_train_split[client_idx] = torch.cat(y_train_split[client_idx])
        y_true_split[client_idx] = torch.cat(y_true_split[client_idx])

    if flag_sort:
        
        # Sort the samples within each client independently, based on the first feature (X[:, 0])
        for client_idx in range(num_clients):
            #sorted_indices = torch.argsort(X_train_split[client_idx][:, 0])
            #sorted_indices = torch.lexsort([X_train_split[client_idx][:, i] for i in range(X_train_split[client_idx].shape[1])])
            sorted_indices = torch.argsort(X_train_split[client_idx], dim=0)
            sorted_indices = sorted_indices[:, 0]  # Retain indices from the first column after sorting

            # Add noise to sorting
            num_swaps = int(len(sorted_indices) * 0.1)  # Swap 10% of indices
            for _ in range(num_swaps):
                i = np.random.randint(0, len(sorted_indices) - swap_range)
                j = i + np.random.randint(1, swap_range + 1)
                sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i].clone()
            
            X_train_split[client_idx] = X_train_split[client_idx][sorted_indices]
            y_train_split[client_idx] = y_train_split[client_idx][sorted_indices]
            y_true_split[client_idx] = y_true_split[client_idx][sorted_indices]

    # Set training labels to zeros
    for client_idx in range(num_clients):
        y_train_split[client_idx] = torch.zeros_like(y_train_split[client_idx])

    # Further split each client's data into rounds
    train_loaders_by_round = [[] for _ in range(num_clients)]
    for client_idx in range(num_clients):
        client_samples = len(X_train_split[client_idx])
        samples_per_round = client_samples // num_rounds
        
        for round_idx in range(num_rounds):
            round_start_idx = round_idx * samples_per_round
            round_end_idx = round_start_idx + samples_per_round if round_idx < num_rounds - 1 else client_samples

            round_dataset = TensorDataset(
                X_train_split[client_idx][round_start_idx:round_end_idx],
                y_train_split[client_idx][round_start_idx:round_end_idx]
            )
            round_loader = DataLoader(round_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            train_loaders_by_round[client_idx].append(round_loader)

    # Prepare the test data (no rounds), full dataset with true labels
    X_test = torch.cat(X_train_split)
    y_test = torch.cat(y_true_split)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Return full data (no rounds) for potential further processing
    all_data = (X, y)
    train_data = (X_train_split, y_train_split)

    return train_loaders_by_round, test_loader, test_dataset, all_data, train_data