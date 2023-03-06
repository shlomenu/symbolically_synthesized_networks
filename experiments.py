import os
import json

import numpy as np
import matplotlib.pyplot as plt

from raven import (
    correctness,
    constant_color,
    all_patterns,
    CORRECTNESS_TARGET_DIM,
    CONSTANT_COLOR_TARGET_DIM,
    ALL_PATTERNS_TARGET_DIM,
    random_split,
    save_split,
    load_split)
from network import baseline
from symbolically_synthesized_networks import SymbolicallySynthesizedNetworks


def run_baseline(
        root_name,
        n_runs,
        train,
        eval,
        target_dim,
        batch_size=8,
        training_iterations=15,
        epochs_per_training_iteration=16,
        perf_metric="acc",
        smoothing_factor=.25,
        use_scheduler=True,
        input_size=128,
        downsampled_size=8,
        vit_dim=256,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=128,
        vit_mlp_dim=512,
        input_channels=1,
        conv_depth=3,
        device="cuda:0"):
    train_runs, eval_runs = [], []
    for i in range(n_runs):
        model = baseline(
            target_dim,
            input_size,
            downsampled_size,
            vit_dim,
            vit_depth,
            vit_heads,
            vit_head_dim,
            vit_mlp_dim,
            input_channels,
            conv_depth,
            device)
        nn_logs = [
            l for _, l in model.run_with_split(
                train,
                eval,
                batch_size,
                training_iterations,
                epochs_per_training_iteration,
                perf_metric,
                smoothing_factor,
                device,
                use_scheduler)]
        with open(os.path.join("results", f"{root_name}_{i}.json"), "w") as f:
            json.dump(nn_logs, f)
        train_runs.append([log["train"]["perf"] for log in nn_logs])
        eval_runs.append([log["eval"]["perf"] for log in nn_logs])
    stat = {
        "perf_metric": perf_metric,
        "avg_train_perf": np.mean(train_runs, axis=0).tolist(),
        "avg_eval_perf": np.mean(eval_runs, axis=0).tolist()}
    with open(os.path.join("results", f"{root_name}_stat.json"), "w") as f:
        json.dump(stat, f)
    return stat


def run_experiment(
        root_name,
        n_runs,
        iterations,
        train,
        eval,
        target_dim,
        max_conn=10,
        device="cuda:0",
        batch_size=8,
        training_iterations=1,
        epochs_per_training_iteration=16,
        perf_metric="acc",
        smoothing_factor=.25,
        use_scheduler=True,
        root_dsl_name="dsl",
        n_retained=50,
        n_preserved=25,
        exploration_timeout=15.,
        exploration_eval_timeout=.1,
        exploration_eval_attempts=1,
        exploration_max_diff=.5,
        exploration_program_size_limit=150,
        compression_iterations=3,
        compression_beta_inversions=2,
        compression_threads=4,
        compression_verbose=True,
        network_input_size=128,
        network_downsampled_size=8,
        network_vit_dim=256,
        network_vit_depth=2,
        network_vit_heads=4,
        network_vit_head_dim=128,
        network_mlp_dim=512,
        network_input_channels=1,
        network_conv_depth=3,
        network_gnn_depth=3,
        network_graph_dim=512,
        network_dropout_rate=.3):
    ssn_logs = []
    ssn = SymbolicallySynthesizedNetworks("graph", "dsl_0")
    for i in range(n_runs):
        ssn_log = []
        for _ in range(iterations):
            ssn_log.append(ssn.run(
                train,
                eval,
                target_dim,
                max_conn,
                device,
                batch_size,
                training_iterations,
                epochs_per_training_iteration,
                perf_metric,
                smoothing_factor,
                use_scheduler,
                root_dsl_name,
                n_retained,
                n_preserved,
                exploration_timeout,
                exploration_eval_timeout,
                exploration_eval_attempts,
                exploration_max_diff,
                exploration_program_size_limit,
                compression_iterations,
                compression_beta_inversions,
                compression_threads,
                compression_verbose,
                network_input_size,
                network_downsampled_size,
                network_vit_dim,
                network_vit_depth,
                network_vit_heads,
                network_vit_head_dim,
                network_mlp_dim,
                network_input_channels,
                network_conv_depth,
                network_gnn_depth,
                network_graph_dim,
                network_dropout_rate))
        dest = os.path.join("results", f"{root_name}_{i}")
        os.mkdir(dest)
        ssn.save_state(dest)
        with open(os.path.join(dest, "log.json"), "w") as f:
            json.dump(ssn_log, f)
        ssn_logs.append(ssn_log)
        ssn.clear_dsls()
        ssn.clear_representations()
        ssn.clear_visualizations()
        ssn.clear_discards()


def load_single_iter_baseline_results(log_base_name, population_size):
    baseline = []
    for i in range(population_size):
        with open(f"results/{log_base_name}_{i}.json") as f:
            log = json.load(f)
        baseline.append((log[0]["train"]["perf"], log[0]["eval"]["perf"]))
    baseline.sort(reverse=True, key=lambda x: x[0])
    return [x for x, _ in baseline], [x for _, x in baseline]


def plot_single_iter_baseline_results(log_base_name, population_size, window_size):
    assert (window_size % 2) == 0
    baseline_train, baseline_eval = load_single_iter_baseline_results(
        log_base_name, population_size)
    assert len(baseline_train) == len(baseline_eval)
    plt.clf()
    plt.plot(range(len(baseline_train)), baseline_train,
             "--b", label="training")
    plt.plot(range(len(baseline_eval)), baseline_eval,
             "--g", label="validation")
    smoothed_eval = moving_avg(baseline_eval, window_size)
    plt.plot(range(window_size // 2, len(smoothed_eval) + (window_size // 2)),
             smoothed_eval, "k", label="validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Rank (Descending)")
    plt.title("Baseline")
    plt.grid(axis="y", color=".95")
    plt.legend(title="Dataset")
    plt.savefig(f"results/single_iteration_baseline.pdf")
    plt.clf()


def load_experimental_results(log_dir):
    with open(f"results/{log_dir}/log.json") as f:
        log = json.load(f)
    trains, evals = [], []
    for round in log:
        perf = [(l[0]["train"]["perf"], l[0]["eval"]["perf"])
                for l in round["nn_training"]]
        perf.sort(reverse=True, key=lambda x: x[0])
        trains.append([x for x, _ in perf])
        evals.append([x for _, x in perf])
    return trains, evals


def plot_experimental_results(log_dir, window_size):
    exp_train, exp_eval = load_experimental_results(log_dir)
    assert (window_size % 2) == 0
    plt.clf()
    for i, (ts, vs) in enumerate(zip(exp_train, exp_eval), start=1):
        assert len(ts) == len(vs)
        plt.plot(range(len(ts)), ts, "--b", label="training")
        plt.plot(range(len(vs)), vs, "--g", label="validation")
        if len(vs) > 15:
            smoothed_vs = moving_avg(vs, window_size)
            plt.plot(range(window_size // 2, len(smoothed_vs) + (window_size // 2)),
                     smoothed_vs, "k", label="validation")
        plt.ylabel("Accuracy")
        plt.xlabel("Rank (Descending)")
        plt.title(f"Experimental (Iteration {i})")
        plt.grid(axis="y", color=".95")
        plt.legend(title="Dataset")
        plt.savefig(f"results/{log_dir}/experimental_iteration_{i}.pdf")
        plt.clf()


def moving_avg(arr, window_size):
    return np.convolve(arr, np.ones((window_size,)), mode="valid") / window_size
