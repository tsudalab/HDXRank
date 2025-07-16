"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
HDXRank main function for running the pipeline.
"""
import argparse
from HDXRank_prot_embedding import BatchTable_embedding, run_embedding
from HDXRank_graph_tasks import save_graphs
from HDXRank_utils import parse_task
from HDXRank_dataset import pepGraph
from HDXRank_prediction import HDXRank_prediction
from HDXRank_score import run_scoring
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger().setLevel(logging.INFO) #can block logging.info by setting to logging.WARNING

def print_hdxrank_info():
    """
    Prints the HDXRank logo and paper information to the screen.
    """
    logo = """
    *****************************************
    *             HDXRank Tool             *
    *****************************************
    """
    paper_info = """
    HDXRank: A tool for applying HDX-MS restraints to protein-protein complex prediction ranking.

    Citation:
    Liyao W, Andrejs T, Songting D, Koji T, Adnan S, "HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data",
    JCTC, 2025. [https://doi.org/10.1021/acs.jctc.5c00175]
    """

    print(logo)
    print(paper_info)
    logging.info("HDXRank initialized. Ready to process data.")

def main():
    print_hdxrank_info()

    parser = argparse.ArgumentParser(description="HDXRank prediction pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the master YAML config file.")
    args = parser.parse_args()

    tasks = parse_task(args.config)

    # tasks checking
    # code here
    ##############################
    tasks['structure_list'] = tasks['structure_list'][:10]
    if tasks.get("TaskParameters", {}).get("Switch", False) in [True, "True", "true", "1"]:
        # Protein Embedding
        print('\n')
        logging.info("Protein Embedding...")
        if tasks['GeneralParameters']['Mode'].lower() == 'train':
            BatchTable_embedding(tasks=tasks)
        elif tasks['GeneralParameters']['Mode'].lower() == 'predict':
            run_embedding(tasks=tasks)
        else:
            raise ValueError(f"Invalid mode: {tasks['GeneralParameters']['Mode']}")

        # Peptide Graph Construction
        print('\n')
        logging.info("Peptide Graph Construction...")
        graph_dataset = pepGraph(
            files=tasks['structure_list'],
            tasks=tasks,
        )
        save_graphs(graph_dataset, tasks["GeneralParameters"]["pepGraphDir"])
    else:
        print('\n')
        logging.info("Skipping Protein Embedding and Peptide Graph Construction")

    # Prediction
    if tasks.get("PredictionParameters", {}).get("Switch", False) in [True, "True", "true", "1"]:
        print('\n')
        logging.info("HDXRank Prediction...")
        HDXRank_prediction(tasks)
    else:
        print('\n')
        logging.info("Skipping HDXRank Prediction")

    # Scoring
    if tasks.get("ScorerParameters", {}).get("Switch", False) in [True, "True", "true", "1"]:
        print('\n')
        logging.info("HDXRank Scoring...")
        run_scoring(tasks)
    else:
        print('\n')
        logging.info("Skipping HDXRank Scoring")

if __name__ == '__main__':
    main()