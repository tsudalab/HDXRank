"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
Modular training pipeline for GearNet model
"""
import os
import torch
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn
from torchdrug.data import DataLoader
from GearNet_revise import GearNet
from HDXRank_utils import parse_task
import random

def set_random_seed(seed=42, deterministic=True, benchmark=False):
    """
    Set a fixed random seed for reproducibility across Python's random module, NumPy, 
    PyTorch (both CPU and CUDA), and scikit-learn.

    Args:
        seed (int): The random seed to use.
        deterministic (bool): If True, ensures deterministic behavior in PyTorch.
        benchmark (bool): If False, disables cuDNN benchmarking for deterministic results.
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure PyTorch deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark  # False for full reproducibility

    # Additional reproducibility settings (useful for some cases)
    #torch.use_deterministic_algorithms(deterministic)

    print(f"Random seed set to {seed}")

# Utility functions
def load_data(tasks):
    """
    Load and preprocess data.

    Args:
        tasks (dict): Parsed XML file.

    Returns:
        tuple: apo_input, complex_input
    """
    summary_HDX_file = os.path.join(tasks["GeneralParameters"]["RootDir"], f"{tasks['GeneralParameters']['TaskFile']}.xlsx")
    hdx_df = pd.read_excel(summary_HDX_file, sheet_name='Sheet1')
    hdx_df = hdx_df.dropna(subset=['structure_file']).drop_duplicates(subset=['structure_file'])

    pepGraph_dirs = tasks['GeneralParameters']['pepGraphDir']
    if not isinstance(pepGraph_dirs, list):
        pepGraph_dirs = [pepGraph_dirs]
    apo_input, complex_input = [], []

    logging.info('Loading data...')
    for pepGraph_dir in pepGraph_dirs:
        print(pepGraph_dir)
        for _, row in tqdm(hdx_df.iterrows(), total=len(hdx_df)):
            pdb = row['structure_file'].strip().split('.')[0].upper()
            pepGraph_file = os.path.join(pepGraph_dir, f'{pdb}.pt')

            if os.path.isfile(pepGraph_file):
                pepGraph_ensemble = torch.load(pepGraph_file)
                if row['complex_state'] == 'single':
                    apo_input.append(pepGraph_ensemble)
                else:
                    complex_input.append(pepGraph_ensemble)

    logging.info(f"Length of apo data: {len(apo_input)}")
    logging.info(f"Length of complex data: {len(complex_input)}")

    return apo_input, complex_input

def prepare_model(input_dim, hidden_dims, num_relation, device, set_scheduler=None,t_max=50):
    """
    Prepare the model and optimizer.

    Args:
        input_dim (int): Input dimension.
        hidden_dims (list): Hidden dimensions for the model.
        num_relation (int): Number of relations.
        device (torch.device): Device to use.

    Returns:
        tuple: model, optimizer, loss_fn
    """
    model = GearNet(input_dim=input_dim, hidden_dims=hidden_dims, num_relation=num_relation,
                    batch_norm=True, concat_hidden=True, readout='sum', activation='relu', short_cut=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

    if set_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif set_scheduler == "cosine":    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    else:
        scheduler = None
    loss_fn = nn.BCELoss()
    return model, optimizer, scheduler, loss_fn

def train_model(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, num_epochs, save_dir, repeat_idx):
    """
    Train the model.

    Args:
        model: PyTorch model instance.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        loss_fn: Loss function instance.
        train_loader: DataLoader for training data.
        device: Device for computation.
        num_epochs: Number of epochs.

    Returns:
        tuple: rmse_train_list, rp_train
    """
    rp_train, rmse_train_list = [], []

    for epoch in range(num_epochs):
        model.train()
        list1_train, list2_train = np.array([]), np.array([])
        epoch_train_losses = []

        for graph_batch in train_loader:
            graph_batch = graph_batch.to(device)
            targets = graph_batch.y
            node_feat = graph_batch.residue_feature.float()
            outputs = model(graph_batch, node_feat)

            train_loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_losses.append(train_loss.item())
            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            #list1_train.extend(target_np)
            #list2_train.extend(output_np)
            list1_train = np.vstack([list1_train, targets]) if not list1_train.size == 0 else targets
            list2_train = np.vstack([list2_train, outputs]) if not list2_train.size == 0 else outputs
            
        if scheduler:
            scheduler.step()
        epoch_rp_train = 0
        for i in range(list1_train.shape[1]):
            epoch_rp_train += np.corrcoef(list2_train[:,i], list1_train[:,i])[0, 1]
        epoch_rp_train /= list1_train.shape[1]

        epoch_train_loss = np.mean(epoch_train_losses)
        #epoch_rp_train = np.mean([np.corrcoef(list2_train[:][i], list1_train[:][i])[0, 1] for i in range(list1_train.shape[1])])
        rp_train.append(epoch_rp_train)
        y = np.array(list1_train).reshape(-1, 1)
        x = np.array(list2_train).reshape(-1, 1)
        epoch_train_rmse = np.sqrt(((y - x) ** 2).mean())
        rmse_train_list.append(epoch_train_rmse)

        if val_loader:
            model.eval()
            epoch_val_losses = []
            list1_val, list2_val = np.array([]), np.array([])

            with torch.no_grad():
                for graph_batch in val_loader:
                    graph_batch = graph_batch.to(device)
                    targets = graph_batch.y
                    node_feat = graph_batch.residue_feature.float()
                    outputs = model(graph_batch, node_feat)

                    val_loss = loss_fn(outputs, targets)
                    epoch_val_losses.append(val_loss.item())

                    targets = targets.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    list1_val = np.vstack([list1_val, targets]) if not list1_val.size == 0 else targets
                    list2_val = np.vstack([list2_val, outputs]) if not list2_val.size == 0 else outputs

            epoch_val_loss = np.mean(epoch_val_losses)
            epoch_val_rmse = np.sqrt(((list1_val - list2_val) ** 2).mean())

        if val_loader:
            logging.info(f'Epoch {epoch}: Loss {epoch_train_loss:.3f}, rho {epoch_rp_train:.3f}, RMSE {epoch_train_rmse:.3f}, val_loss {epoch_val_loss:.3f}, val_RMSE {epoch_val_rmse:.3f}')
        else:
            logging.info(f'Epoch {epoch}: Loss {epoch_train_loss:.3f}, rho {epoch_rp_train:.3f}, RMSE {epoch_train_rmse:.3f}')

        #if epoch % 5 == 0:
        #    save_checkpoint(model, optimizer, epoch, os.path.join(save_dir, f'HDXRank_epoch{epoch}_v{repeat_idx}.pth'))

    return rmse_train_list, rp_train

def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)

def main():
    parser = argparse.ArgumentParser(description='Train new HDXRank model.')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--save', type=str, default=None, help='Directory to save the model (overrides config)')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--cuda', type=int, default=None, help='CUDA device number (overrides config)')
    parser.add_argument('--train_val_split', type=float, default=None, help='Proportion of data to use for validation (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training (overrides config)')
    parser.add_argument('--repeat', type=int, default=None, help='Repeat training process (overrides config)')
    parser.add_argument('--name', type=str, default=None, help='Model prefix (overrides config)')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    #FIXME: remove keys usage
    keys, tasks = parse_task(args.config)
    train_cfg = tasks.get('TrainParameters', {})

    # Allow CLI overrides
    save_dir = args.save if args.save else train_cfg.get('SaveDir', './models')
    num_epochs = args.epoch if args.epoch is not None else train_cfg.get('Epochs', 100)
    cuda_device = args.cuda if args.cuda is not None else train_cfg.get('CudaID', 0)
    train_val_split = args.train_val_split if args.train_val_split is not None else train_cfg.get('TrainValSplit', 0.2)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get('BatchSize', 16)
    repeat = args.repeat if args.repeat is not None else train_cfg.get('Repeat', 1)
    model_name = args.name if args.name else train_cfg.get('ModelName', 'HDXRank_')

    seeds = train_cfg.get('Seeds', [42, 43, 44, 45, 46])
    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

    apo_input, complex_input = load_data(tasks)

    for i in range(repeat):
        set_random_seed(seed=seeds[i])
        train_apo, val_apo = train_test_split(apo_input, test_size=train_val_split, random_state=seeds[i])
        train_complex, val_complex = train_test_split(complex_input, test_size=train_val_split, random_state=seeds[i])

        # flatten the list
        train_apo = [item for sublist in train_apo for item in sublist]
        train_complex = [item for sublist in train_complex for item in sublist]
        val_apo = [item for sublist in val_apo for item in sublist]
        val_complex = [item for sublist in val_complex for item in sublist]

        train_set = train_apo + train_complex
        val_set = val_apo + val_complex
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        logging.info(f'training set: {len(train_set)}')
        logging.info(f'validation set: {len(val_set)}')

        input_dim = train_cfg.get('InputDim', 56)
        hidden_dims = train_cfg.get('HiddenDims', [64, 64, 64])
        model, optimizer, scheduler, loss_fn = prepare_model(input_dim=input_dim, hidden_dims=hidden_dims, num_relation=7, 
                                                             device=device, set_scheduler="cosine", t_max=100)
        rmse_train_list, rp_train = train_model(model, optimizer, scheduler, loss_fn, train_loader, val_loader, device, num_epochs, save_dir, i)

        model_file = f'{model_name}_epoch{num_epochs}_v{i}.pth'
        save_checkpoint(model, optimizer, num_epochs, os.path.join(save_dir, model_file))

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    main()
