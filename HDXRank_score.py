import os
import argparse
import pandas as pd
from tqdm import tqdm
from HDXRank_utils import Scorer, parse_task

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_scoring(tasks):
    """
    Calculates and saves HDXRank scores based on the provided configuration.
    
    Args:
        tasks (dict): The main configuration dictionary loaded from YAML.
    """
    # Get scorer-specific settings and protein states
    settings, apo_states, complex_states = Scorer.parse_config(tasks)
    logger.info("Successfully parsed scorer configuration.")
    logger.debug(f"Apo states: {apo_states}")
    logger.debug(f"Complex states: {complex_states}")

    ## calculate HDXRank Score ###
    general_params = tasks.get('GeneralParameters', {})
    pred_params = tasks.get('PredictionParameters', {})
    task_params = tasks.get('TaskParameters', {})

    root_dir = general_params.get('RootDir', '.')
    save_dir = settings.get('save_dir', '.')
    protein_name = settings.get('protein_name', task_params.get('Protein')) 
    cluster_id = settings['cluster_id']
    HDX_fpath = os.path.join(root_dir, settings["HDX_file"])
    pred_cluster = settings['pred_cluster']
    timepoints = settings['timepoints']

    hdx_true_diffs = []
    hdx_epitope_peps = []
    for apo, complex_ in zip(apo_states, complex_states):
        logger.info(f"Processing apo: {apo}, complex: {complex_}")
        true_diff, _ = Scorer.get_true_diff(HDX_fpath, apo, complex_, cluster_id, timepoints)
        _, epitope_pep = Scorer.get_hdx_epitopes(true_diff)
        hdx_true_diffs.append(true_diff)
        hdx_epitope_peps.append(epitope_pep)

    pred_suffix = settings.get('pred_suffix', '')
    pred_dir = pred_params.get('PredDir')
    pred_df = Scorer.parse_predictions(pred_dir, suffix=pred_suffix)

    if pred_df.empty:
        logger.error(f"No prediction files found in {pred_dir} with suffix '{pred_suffix}'. Exiting.")
        return

    logger.debug(f"Prediction data head:\n{pred_df.head()}")
    logger.info(f"Number of unique batches: {len(pred_df['Batch'].unique())}")

    pred_df_dict = {batch: group for batch, group in pred_df.groupby('Batch')}
    complex_batch_list = [batch for batch in pred_df_dict.keys() if 'MODEL' in batch]
    logger.info(f"Found {len(complex_batch_list)} model batches to score.")

    HDX_scores = {}
    y_true_list, y_pred_list = [], []
    for complex_batch in tqdm(complex_batch_list, desc="Scoring models"):
        y_true, y_pred = Scorer.prepare_data(
            pred_df_dict, complex_batch, apo_states, hdx_true_diffs, 
            hdx_epitope_peps=hdx_epitope_peps, pred_cluster=pred_cluster
        )
        if y_true is None or len(y_true) == 0:
            logger.warning(f"No common peptides found for batch {complex_batch}. Skipping.")
            continue
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        HDX_scores[complex_batch] = {
            'batch': complex_batch, 
            'HDXRank_score': Scorer.root_mean_square_error(y_true, y_pred)
        }

    if not HDX_scores:
        logger.error("No scores were calculated. Please check your prediction files and configuration.")
    else:
        score_df = pd.DataFrame(HDX_scores).T.reset_index(drop=True)
        score_df = score_df.sort_values(by='HDXRank_score', ascending=True)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{protein_name}_HDXRank_score.csv')
        score_df.to_csv(save_path, index=False)
        logger.info(f"HDXRank score saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate HDXRank scores based on predictions.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--save', default=None, help='(Optional) Directory to save score file. Overrides config setting.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    _, tasks = parse_task(args.config)
    
    # Allow command-line --save to override the config setting
    if args.save:
        if 'ScorerSettings' not in tasks: tasks['ScorerSettings'] = {}
        if 'settings' not in tasks['ScorerSettings']: tasks['ScorerSettings']['settings'] = {}
        tasks['ScorerSettings']['settings']['save_dir'] = args.save
        
    run_scoring(tasks)

if __name__ == '__main__':
    main()
