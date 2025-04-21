from utils.rttm import csvs_to_rttms
from argparse import ArgumentParser
from pathlib import Path
from calculate_metrics import main as rttms_to_metrics
from run_failure_analysis import main as rttms_to_visualizations


def main(model):
    PREDICTIONS_ROOT = "predictions"
    # CSV to RTTM
    input_dir = Path(PREDICTIONS_ROOT) / f"{model}_csvs"
    output_dir = Path(PREDICTIONS_ROOT) / f"{model}_rttms"
    csvs_to_rttms(input_dir=input_dir, output_dir=output_dir)
    # RTTM to Metrics
    rttms_to_metrics()
    # Visualizations
    rttms_to_visualizations()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-m", dest="model", type=str)
    args = argparser.parse_args()
    main(model=args.model)
