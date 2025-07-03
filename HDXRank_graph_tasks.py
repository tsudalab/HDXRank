"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
Generates and saves peptide-level graphs from protein embeddings for model input.
"""
import os
import argparse
import torch
from HDXRank_dataset import pepGraph
from HDXRank_utils import parse_task
from tqdm import tqdm
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def save_graphs(graph_dataset, save_dir):
    """
    Saves the generated graphs to the specified directory.

    Args:
        graph_dataset (pepGraph): A dataset object containing the generated graphs.
        save_dir (str): The directory where the graphs will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    logger.info(f"Saving graphs to {save_dir}...")
    for i, data in tqdm(enumerate(graph_dataset), total=len(graph_dataset), desc="Saving Graphs"):
        if data is None:
            continue
        graph_ensemble, label = data
        path = os.path.join(save_dir, f'{label}.pt')
        if len(graph_ensemble) == 0:
            logger.warning(f"Empty graph ensemble for label {label}, skipping.")
            continue
        torch.save(graph_ensemble, path)
        count += len(graph_ensemble)
    logger.info(f"Successfully saved {count} graphs to {save_dir}")

def main():
    """Main function to run the graph generation script."""
    logger.info("Running graph generation as a standalone script.")
    parser = argparse.ArgumentParser(description='Generate and save peptide graphs from embeddings.')
    parser.add_argument('--config', type=str, required=True, help='Path to the master YAML config file.')
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    keys, tasks = parse_task(args.config)

    logger.info("Initializing graph dataset...")
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
    logger.info("Standalone graph generation script finished.")

if __name__ == "__main__":
    main()
