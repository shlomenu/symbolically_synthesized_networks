# Graph-based Symbolically Synthesized Neural Networks (G-SSNNs)

This is the main project repository for the paper [Symbolically Synthesized Neural Networks](https://arxiv.org/abs/2303.03340):

> **Abstract**: Neural networks adapt very well to distributed and continuous representations, but struggle to learn and generalize from small amounts of data. Symbolic systems commonly achieve data efficient generalization by exploiting modularity to benefit from local and discrete features of a representation. These features allow symbolic programs to be improved one module at a time and to experience combinatorial growth in the values they can successfully process. However, it is difficult to design components that can be used to form symbolic abstractions and which are highly-overparametrized like neural networks, as the adjustment of parameters makes the semantics of modules unstable. I present Graph-based Symbolically Synthesized Neural Networks (G-SSNNs), a form of neural network whose topology and parameters are informed by the output of a symbolic program. I demonstrate that by developing symbolic abstractions at a population level, and applying gradient-based optimization to such neural models at an individual level, I can elicit reliable patterns of improved generalization with small quantities of data known to contain local and discrete features. The paradigm embodied by G-SSNNs offers a route towards the communal development of compact and composable abstractions which can be flexibly repurposed for a variety of tasks and high-dimensional media. In future work, I hope to pursue these benefits by exploring more ambitious G-SSNN designs based on more complex classes of symbolic programs. The code and data associated with the reported results are publicly available at this https URL .

This repository binds together the data generation capabilities offered by [`raven-gen`](https://pypi.org/project/raven-gen/), the library learning capabilities of [`stitch-core`](https://pypi.org/project/stitch-core/), the distributional program search and file management capabilities of [`antireduce`](https://github.com/shlomenu/antireduce), and the relational structures defined by [`antireduce-graphs`](https://github.com/shlomenu/antireduce-graphs) to support training and evaluation of G-SSNNs.  

## Module/Directory reference

### `graph/`

This directory exists after step (1) below has been performed.  The `graph/` directory contains the subdirectories used by OCaml binaries and the `SymbolicallySynthesizedNetworks` class to manage the state of a population of G-SSNNs.  This includes tracking discarded members of the population through `graph/discards`, tracking the current population through `graph/representations`, tracking iterations of the current Domain-Specific Library (DSL) through `graph/dsls`, and tracking visualizations of the current population through `graph/visualizations`.  This directory also contains the dataset synthesized and used in official experiments with the functions of the `raven.py` module.  The file `graph/128_split_correct.pkl` captures the split of a random subset of the data that was used for training and validation in official experiments.

### `program_synthesis/`

This directory contains OCaml executables which can be compiled by running `make` from the root of the repository, or by running `dune build` from within `program_synthesis/`, provided that steps (2-4) have been performed.  After running one of these commands, appropriate `*.exe` files (this naming is platform independent) binaries should exist within `program_synthesis/_build/default`.  The `utilities.py` module wraps calls to execute these binaries in ordinary Python functions.

### `results/`

The results directory functions as an archive of current experimental results.  At the root level of the `results/` directory exist files corresponding to baseline runs.  There are currently three runs, only one of which is directly relevant to analysis of the performance of G-SSNNs.  The baseline from the paper is contained in the files named `128_const_color_1_iteration_baseline_*`.  The baseline plot from the paper is included here too as `128_const_color_1_iteration_baseline_plot.pdf`; the file was renamed since official submission but is otherwise unaltered.  The other files at the root of the repository contain logs of baseline runs that took more than one iteration; the other `128_const_color_*_iteration_baseline_summary.json` files contain data on the average performance of a population of 10 baseline models across 10 and 15 iterations.  

The subdirectory `results/ssn_0_0` archives the results of the run of the experimental setting reported in the paper.  It includes the contents of `graph/{discards,dsls,representations,visualizations}` at the time the run concluded, as well as a complete `log.json` containing the experimental data and a `discard_hist.json`, showing which files in `results/ssn_0_0/discards` were removed after which iterations.  The plots used in the paper to depict results in the experimental setting are present in this subdirectory as well.

### `utilities.py`

Contains python wrapper functions around calls to executables resulting from OCaml compilation.  The three currently used functionalities from this module are given by `explore`, `stitch_compress`, and `incorporate_stitch`.  The first of these performs distributional program search in the style of the `antireduce` and `antireduce-graphs` libraries; the second performs library learning by delegating to the compression functionality of the `stitch_core` package; and the third performs work needed to update artifacts in `graph/dsls` and `graph/representations` that are affected by the creation of primitives and rewriting of programs performed via `stitch_core`.

### `raven.py`

Contains functions for generating data with `raven-gen`, loading this data as a subclass of `torch.utils.data.dataset.Dataset`, producing random two way splits of this data, and extracting different ground truth annotations.   

### `network.py`

Contains classes representing baseline and experimental models together with shared training routines.  The class `GraphStructured` provides the implementation of the embedding function described in the G-SSNNs section of the paper. 

### `symbolically_synthesized_networks.py`

Contains the class `SymbolicallySynthesizedNetworks` which evolves a population of G-SSNNs by iteratively performing program synthesis, training a G-SSNN for each program in the generated population, then discarding the bottom half of programs according to training accuracy.

### `experiments.py`

Provides high-level routines for running the experiments and baselines reported in the experiment.  The `run_*` methods of this module also archive their results within the `results/` directory.  The separate `plot_*` methods must be used in order to generate accompanying plots.

## Instructions for use

1. Unzip `graph.zip` to recreate the intended `graph/` directory structure.

2. Install OPAM ([guide](https://opam.ocaml.org/doc/Install.html)) and create a switch `opam switch create 4.14.0`.

3. Clone the [`antireduce`](https://github.com/shlomenu/antireduce) and [`antireduce-graphs`](https://github.com/shlomenu/antireduce-graphs) repositories.  From within the root of each, run `opam install .` with the previously-created switch.  

4. Import the remaining configuration to the switch with the command `opam switch import opam-switch.freeze`.  If this does not succeed, attempt to install the packages listed under `roots` in the file `opam-switch.freeze`.  

5. Create a Python virtual environment and install the following packages; for the precise versions used in official experiments, consult `requirements.txt`:
 - `raven-gen`
 - `matplotlib`
 - `numpy`
 - `tqdm`
 - `dgl`
 - `torch`
 - `torchvision`
 - `vit-pytorch`
 - `einops`
 - `pygraphviz`
 - `stitch_core`

6. Generate data in an interpreter with the following commands:

```
>>> from raven import *
>>> generate_data(20000, "graph/dataset", save_pickle=True)
>>> train, eval = random_split("graph/dataset", constant_color(True), constant_color(False), 128, batch_size=8, n_eval_batches=150, include_incorrect=False)
>>> save_split("graph/128_split_correct.pkl")
```

7. Run the baseline and generate baseline plots with the following commands:

```
>>> from experiments import *
>>> train, eval = load_split("graph/128_split_correct.pkl", "graph/dataset", constant_color(True), constant_color(False), 128, batch_size=8, n_eval_batches=150, include_incorrect=False)
>>> run_baseline("128_const_color_1_iteration_baseline", 50, train, eval, CONSTANT_COLOR_TARGET_DIM, training_iterations=1)
>>> plot_single_iter_baseline_results("128_const_color_1_iteration_baseline", 50, 6)
```

8. Run the experimental models with the following commands:

```
>>> from experiments import *
>>> train, eval = load_split("graph/128_split_correct.pkl", "graph/dataset", constant_color(True), constant_color(False), 128, batch_size=8, n_eval_batches=150, include_incorrect=False)
>>> run_experiment("ssn_0", 1, 5, train, eval, CONSTANT_COLOR_TARGET_DIM)
>>> plot_experimental_results("ssn_0_0", 6)
```