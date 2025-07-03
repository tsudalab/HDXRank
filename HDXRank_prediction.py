"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
HDXRank prediction pipeline.
"""
import os
import argparse
import logging

import torch
from torchdrug import data

import pandas as pd
import numpy as np
from GearNet_revise import GearNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def test_model(model, test_loader, device):
    y_pred = []
    y_true = []
    range_list = []
    chain_list = []
    model.eval()
    try:
        with torch.no_grad():
            for i, graph_batch in enumerate(test_loader):
                    graph_batch = graph_batch.to(device)
                    targets = graph_batch.y
                    node_feat = graph_batch.residue_feature.float()
                    outputs = model(graph_batch, node_feat)#[:,1] # revised model output 3 values for each cluster
                    range_list.extend(graph_batch.range.cpu().detach().numpy())
                    chain_list.extend(graph_batch.chain.cpu().detach().numpy())
                    y_pred.append(outputs.cpu().detach().numpy())
                    y_true.append(targets.cpu().detach().numpy())
            y_pred = np.concatenate(y_pred, axis=0) if len(y_pred) > 0 else []
            y_true = np.concatenate(y_true, axis=0) if len(y_true) > 0 else []
    except Exception as e:
        logging.error(e)
        return None, None, None, None

    return y_true, y_pred, range_list, chain_list

def load_data(pepGraph_dir, keys, batch_size=64):
    graph_list = [graph.strip().split('.')[0].upper() for graph in keys[3]]
    batch_list = []
    input_data = []
    for i, graph in enumerate(graph_list):
        load_fpath = os.path.join(pepGraph_dir, f'{graph}.pt')
        if not os.path.exists(load_fpath):
            #logging.error(f'Missing file: {load_fpath}')
            continue
        graph_data = torch.load(load_fpath)
        input_data.extend(graph_data)
        batch_list.extend([graph] * len(graph_data))
    logging.info(f'total test data:{len(input_data)}')
    return data.DataLoader(input_data, batch_size=batch_size, shuffle=False), np.array(batch_list)

def HDXRank_prediction(tasks, keys):
    pepGraph_dir = tasks["GeneralParameters"]["pepGraphDir"]
    model_list = [os.path.join(tasks["PredictionParameters"]["ModelDir"], f'{model_name}.pth') for model_name in tasks["PredictionParameters"]["ModelList"]]
    device = torch.device(f"cuda:{tasks['PredictionParameters']['CudaID']}" if torch.cuda.is_available() else 'cpu')

    # model loading
    for i, model_path in enumerate(model_list):
        model = GearNet(input_dim = 56, hidden_dims = [64,64,64],
                        num_relation=7, batch_norm=True, concat_hidden=True, readout='sum', activation = 'relu', short_cut=True)
        model_state_dict = torch.load(model_path, map_location=device)
        model_state_dict = model_state_dict['model_state_dict']
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        logging.info('model loaded successfully!')

        # load data
        Pred_dataloader, batch_list = load_data(pepGraph_dir, keys, batch_size=tasks["PredictionParameters"]["BatchSize"])

        # prediction
        y_true_list, y_pred_list, range_list, chain_list = test_model(model, Pred_dataloader, device)
        range_list = np.array(range_list).reshape(-1, 2)
        chain_list = np.array(chain_list)
        x_strings = np.array([f'{int(start)}-{int(end)}' for i, (start, end) in enumerate(range_list)])
        y_true_list = np.array(y_true_list)
        y_pred_list = np.array(y_pred_list)

        data = {
        'Batch': batch_list,
        'Y_True_short': y_true_list[:,0],
        'Y_True_middle': y_true_list[:,1],
        'Y_True_long': y_true_list[:,2],
        'Y_Pred_short': y_pred_list[:,0],
        'Y_Pred_middle': y_pred_list[:,1],
        'Y_Pred_long': y_pred_list[:,2],
        'Chain': chain_list,
        'Range': x_strings
        }

        model_name = os.path.basename(model_path).split('.')[0]
        model_name = "HDXRank"+os.path.basename(model_path).split('.')[0][-3:]
        pepG_folder = os.path.basename(pepGraph_dir)
        output_file = f'prediction_{pepG_folder}_{model_name}.csv'
        results_df = pd.DataFrame(data)
        results_csv_path = os.path.join(tasks["PredictionParameters"]["PredDir"], output_file)
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
        results_df.to_csv(results_csv_path, index=False)
        logging.info(f'results saved to csv: {results_csv_path}')

if __name__ == "__main__":
    from HDXRank_utilis import parse_task

    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description='Train new HDXRank model.')
    parser.add_argument('-input', type=str, required=True, help='path to XML task file (require general parameters)')
    args = parser.parse_args()
    keys, tasks = parse_task(args.input)

    HDXRank_prediction(tasks, keys)