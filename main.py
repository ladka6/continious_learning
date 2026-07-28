import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    seed_override = args.seed
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json
    if seed_override is not None:
        # One seed per SLURM array job: --seed overrides the config's list.
        args["seed"] = [seed_override]

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./exps/tosca_ina_ridge_router.json',
                        help='Json file of settings.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this seed (overrides the config seed list).')
    return parser

if __name__ == '__main__':
    main()
