"""Tier 2 hyperparameter grid: unlike lambda/proj_dim, params that shape what
the TOSCA adapters learn (lr, l1, epochs, mlp_ratio, se_ratio, ffn_num, ...)
require a FULL retrain per value -- there is no offline shortcut. This script
only generates the config + sbatch files for a cartesian grid over a base
config; it does NOT submit anything. Review the printed job count (each job
costs roughly what your existing *_ridge_router.json runs cost) before
submitting, and submit only the jobs you actually want to spend GPU-hours on.

Usage:
    python gen_grid_configs.py --base exps/tosca_ina_ridge_router.json \
        --param lr=0.01,0.025,0.05 --param l1=0.0001,0.0005 --tag ina_lr_l1

Writes:
    exps/sweeps/<tag>/<tag>_<param=val>_<param=val>.json
    run_sweeps/<tag>/run_<tag>_<param=val>_<param=val>.sh

Prints the sbatch commands for you to run by hand.
"""
import argparse
import itertools
import json
import os


def parse_value(raw):
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def slugify(param, value):
    s = f"{param}{value}".replace(".", "p").replace("-", "m")
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="base *_ridge_router.json to vary")
    parser.add_argument(
        "--param", action="append", required=True,
        help="name=v1,v2,... ; repeatable for a cartesian grid over multiple params",
    )
    parser.add_argument("--tag", type=str, required=True, help="short name for this sweep")
    parser.add_argument("--partition", type=str, default="gpu_a100")
    parser.add_argument("--time", type=str, default="12:00:00")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--cpus-per-task", type=str, default="8")
    parser.add_argument("--mem", type=str, default="40G")
    cli = parser.parse_args()

    with open(cli.base) as f:
        base_config = json.load(f)

    grid = {}
    for spec in cli.param:
        name, values = spec.split("=", 1)
        grid[name] = [parse_value(v) for v in values.split(",")]

    exps_dir = os.path.join("exps", "sweeps", cli.tag)
    run_dir = os.path.join("run_sweeps", cli.tag)
    os.makedirs(exps_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    names = list(grid.keys())
    value_lists = [grid[n] for n in names]
    combos = list(itertools.product(*value_lists))

    print(f"Base config: {cli.base}")
    print(f"Grid over {names}: {len(combos)} combinations")
    print("Each job trains the full incremental sequence (same cost as your "
          "existing *_ridge_router.json runs) -- nothing here is a shortcut.\n")

    sbatch_cmds = []
    for combo in combos:
        overrides = dict(zip(names, combo))
        config = dict(base_config)
        config.update(overrides)

        combo_slug = "_".join(slugify(n, v) for n, v in overrides.items())
        job_name = f"tosca-{cli.tag}-{combo_slug}"
        config_path = os.path.join(exps_dir, f"{job_name}.json")
        script_path = os.path.join(run_dir, f"run_{job_name}.sh")

        # Checkpoints are namespaced by dataset + prefix (see Learner._ckpt_dir).
        # Grid variants share the base config's dataset, so each needs its own
        # prefix or they'd all read/write the same tosca/<dataset> checkpoints.
        config["prefix"] = job_name

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={cli.partition}
#SBATCH --gpus={cli.gpus}
#SBATCH --cpus-per-task={cli.cpus_per_task}
#SBATCH --mem={cli.mem}
#SBATCH --time={cli.time}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

python main.py --config {config_path}
"""
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        sbatch_cmds.append(f"sbatch {script_path}")
        print(f"  {overrides} -> {config_path}")

    print(f"\n{len(combos)} config(s) + sbatch script(s) written under {exps_dir}/ and {run_dir}/.")
    print("Nothing has been submitted. To launch (all, or pick a subset):\n")
    for cmd in sbatch_cmds:
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
