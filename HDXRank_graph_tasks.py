"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
Generate and save peptide graphs for model training
"""
import os
import pandas as pd
import torch
from HDXRank_dataset import pepGraph
from HDXRank_utilis import parse_task
from tqdm import tqdm
import logging

def save_graphs(graph_dataset, save_dir):
    """
    Save the generated graphs to disk.

    Args:
        graph_dataset (pepGraph): Dataset containing the generated graphs.
        save_dir (str): Directory to save the graphs.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for i, data in tqdm(enumerate(graph_dataset), total=len(graph_dataset)):
        if data is None:
            continue
        graph_ensemble, label = data
        path = f'{save_dir}/{label}.pt'
        if len(graph_ensemble) == 0:
            continue
        torch.save(graph_ensemble, path)
        count += len(graph_ensemble)
    logging.info(f"Saved {count} graphs to {save_dir}")

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    import argparse
    parser = argparse.ArgumentParser(description='Generate protein embeddings.')
    parser.add_argument('-input', type=str, required=True, help='XML task file path')
    args = parser.parse_args()
    keys, tasks = parse_task(args.input)

    graph_dataset = pepGraph(
        keys,
        tasks["GeneralParameters"]["RootDir"],
        tasks["GeneralParameters"]["EmbeddingDir"],
        tasks["GeneralParameters"]["PDBDir"],
        tasks["GeneralParameters"]["pepGraphDir"],
        min_seq_sep=tasks["GraphParameters"]["SeqMin"],
        max_distance=tasks["GraphParameters"]["RadiusMax"],
        graph_type=tasks["GraphParameters"]["GraphType"],
        embedding_type=tasks["GraphParameters"]["EmbeddingType"],
        max_len=tasks["GraphParameters"]["MaxLen"],
        pep_range=tasks["GraphParameters"]["PepRange"]
    )

    save_graphs(graph_dataset, tasks["GeneralParameters"]["pepGraphDir"])
