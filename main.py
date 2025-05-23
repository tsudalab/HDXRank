"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
HDXRank main function for running the pipeline.
"""
import argparse
from HDXRank_prot_embedding import BatchTable_embedding, XML_embedding
from HDXRank_graph_tasks import save_graphs
from HDXRank_utilis import parse_task
from HDXRank_dataset import pepGraph
from HDXRank_prediction import HDXRank_prediction
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
    *        Welcome to use HDXRank         *
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
    parser.add_argument("-input", type=str, required=True, help="Path to XML setting file.")
    args = parser.parse_args()

    keys, tasks = parse_task(args.input)

    # tasks checking
    # code here
    ##############################

    # Protein Embedding
    if tasks["EmbeddingParameters"]["Switch"] == "True":
        print('\n')
        logging.info("Protein Embedding...")
        if tasks['GeneralParameters']['Mode'] == 'BatchTable':
            BatchTable_embedding(tasks = tasks)
        else:
            XML_embedding(tasks = tasks)
    else:
        print('\n')
        logging.info("Skipping Protein Embedding")
    
    # Peptide Graph Construction
    if tasks["TaskParameters"]["Switch"] == "True":
        print('\n')
        logging.info("Peptide Graph Construction...")
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
    else:
        print('\n')
        logging.info("Skipping Peptide Graph Construction")

    # Prediction
    if tasks["PredictionParameters"]["Switch"] == "True":
        print('\n')
        logging.info("HDXRank Prediction...")
        HDXRank_prediction(tasks, keys)
    else:
        print('\n')
        logging.info("Skipping HDXRank Prediction")

if __name__ == '__main__':
    main()