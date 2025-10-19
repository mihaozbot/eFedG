import torch
import threading
from utils.utils_plots import  plot_first_feature_combinations
import matplotlib.pyplot as plt
import os

def dataloader_to_tensors(loader, device=None):
    xs, ys = [], []
    for xb, yb in loader:          # one pass
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs, 0)
    Y = torch.cat(ys, 0)
    if device is not None:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
    return X, Y

def train_supervised(model, data_loader, index, round=0, debug_flag=0):
    
    model.set_debugging_flag(debug_flag)

    for batch, (data, labels) in enumerate(data_loader):
        data, labels = data.to(model.device, non_blocking=True), labels.to(model.device, non_blocking=True)
        model.batch_clustering(data, labels)

        if debug_flag in (8, 9):
            if index == 0:
                title = f"Client {index} • Round {round} • Batch {batch}"
                fig = plot_first_feature_combinations((data, labels), model=model, num_sigma=2, N_max=model.kappa_n, title=title)
                if debug_flag == 9:
                    
                    plt.show()
                    plt.pause(0.001)

                else:
                    outdir = f".Images/Train/Batch/Client_{index}"
                    os.makedirs(outdir, exist_ok=True)
                    base = f"{outdir}/round_{round}_batch_{batch}"
                    fig.savefig(f"{base}.svg", bbox_inches='tight')
                    fig.savefig(f"{base}.png", bbox_inches='tight')
                    fig.savefig(f"{base}.pdf", bbox_inches='tight')

    return model


def train_models_in_threads(models, data_loaders, round=0, debug_flag=0):
    threads = []
    trained_models = {}

    def thread_function(model, data_loader, index, round, debug_flag):
        nonlocal trained_models

        trained_model = train_supervised(model, data_loader, index, round, debug_flag)
        trained_models[index] = trained_model 

    for index, (model, data_loader) in enumerate(zip(models, data_loaders)):
        thread = threading.Thread(target=thread_function, args=(model, data_loader, index, round, debug_flag))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return [trained_models[i] for i in range(len(models))] 
