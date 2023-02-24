import os
import shutil
import json
from collections import defaultdict

import pygraphviz as pgv

from network import program_structured
import utilities


class SymbolicallySynthesizedNetworks:

    def __init__(self,
                 name_of_domain,
                 dsl_name):
        self.name_of_domain = name_of_domain
        self.previous_abstractions = 0
        self.representations_path = os.path.join(
            self.name_of_domain, "representations")
        self.visualizations_path = os.path.join(
            self.name_of_domain, "visualizations")
        self.discards_path = os.path.join(
            self.name_of_domain, "discards")
        self.dsl_path = os.path.join(self.name_of_domain, "dsls")
        self.load_dsl(dsl_name)
        self.representations = sorted((filename for filename in os.listdir(
            self.representations_path) if filename.endswith(".json")))
        self.masses, self.programs = self._fetch_meta()
        self.discards = []

    def run(self,
            train,
            eval,
            target_dim,
            max_conn,
            device,
            batch_size,
            iterations,
            epochs_per_iteration,
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
            network_vit_mlp_dim,
            network_input_channels,
            network_conv_depth,
            network_gnn_depth,
            network_graph_dim,
            network_dropout_rate):
        assert n_preserved < n_retained
        exploration_kwargs = {
            "exploration_timeout": exploration_timeout,
            "program_size_limit": exploration_program_size_limit,
            "eval_timeout": exploration_eval_timeout,
            "eval_attempts": exploration_eval_attempts,
            "max_diff": exploration_max_diff,
            "max_conn": max_conn
        }
        compression_kwargs = {
            "iterations": compression_iterations,
            "n_beta_inversions": compression_beta_inversions,
            "threads": compression_threads,
            "verbose": compression_verbose
        }
        network_kwargs = {
            "input_size": network_input_size,
            "downsampled_size": network_downsampled_size,
            "vit_dim": network_vit_dim,
            "vit_depth": network_vit_depth,
            "vit_heads": network_vit_heads,
            "vit_head_dim": network_vit_head_dim,
            "vit_mlp_dim": network_vit_mlp_dim,
            "input_channels": network_input_channels,
            "conv_depth": network_conv_depth,
            "gnn_depth": network_gnn_depth,
            "graph_dim": network_graph_dim,
            "max_conn": max_conn,
            "dropout_rate": network_dropout_rate
        }
        if self.n_representations == 0:
            print(f"initial exploration...")
            exploration_log = self.explore(
                n_retained,
                next_dsl_name=root_dsl_name,
                **exploration_kwargs)
            print(
                f"\treplaced: {exploration_log['replaced']}\n"
                f"\tmax. novel representations: {exploration_log['max_novel_representations']}\n"
                f"\tnew: {exploration_log['new']}\n"
                f"\ttotal: {exploration_log['total']}\n"
                f"\tmin. mass: {exploration_log['min_mass']}\n"
                f"\tmax. mass: {exploration_log['max_mass']}\n"
                f"\tavg. mass: {exploration_log['avg_mass']}\n"
            )
            exploration_log["iteration"] = 0
            exploration_log["activity"] = "exploration"
        nn_logs = []
        self.clear_visualizations()
        self.visualize()
        for i, r in enumerate(self.representations, start=1):
            structured = program_structured(
                self.load_representation(r)["output"],
                target_dim,
                device=device,
                **network_kwargs)
            nn_log = [
                l for _, l in
                structured.run_with_split(train,
                                          eval,
                                          batch_size,
                                          iterations,
                                          epochs_per_iteration,
                                          perf_metric,
                                          smoothing_factor,
                                          device,
                                          use_scheduler=use_scheduler)]
            nn_logs.append(nn_log)
            print(
                f"{i} of {self.n_representations}:\n"
                f"\tname: {r}\n"
                f"\tmass: {self.masses[r]}\n"
                "\taggregate training state.:\n"
                f"\t\tloss: {nn_log[-1]['training']['final_loss']:.4f}, "
                f"perf.: {nn_log[-1]['training']['perf']:.4f}\n"
                "\taggregate training set stat.:\n"
                f"\t\tloss: {nn_log[-1]['train']['final_loss']:.4f}, "
                f"perf.: {nn_log[-1]['train']['perf']:.4f}\n"
                "\taggregate evaluation set stat.:\n"
                f"\t\tloss: {nn_log[-1]['eval']['final_loss']:.4f}, "
                f"perf.: {nn_log[-1]['eval']['perf']:.4f}"
            )
        performance_ordered = sorted(
            zip(nn_logs, self.representations),
            key=lambda x: x[0][-1]["train"]["perf"],
            reverse=True)
        nn_logs = [l for l, _ in performance_ordered]
        self.representations = [
            r for _, r in performance_ordered[:n_preserved]]
        self.discards.append([r for _, r in performance_ordered[n_preserved:]])
        for d in self.discards[-1]:
            os.replace(os.path.join(self.representations_path, d),
                       os.path.join(self.discards_path, d))
        print("compressing...")
        compression_log = self.compress(
            next_dsl_name=root_dsl_name, **compression_kwargs)
        print(
            f"\nnumber of primitives added during compression: {compression_log['n_added']}")
        if compression_log["n_added"] > 0:
            print(
                f"\tnew dsl mass: {compression_log['next_dsl_mass']}")
        print("exploring...")
        exploration_log = self.explore(
            n_retained,
            next_dsl_name=root_dsl_name,
            **exploration_kwargs)
        print(
            f"\treplaced: {exploration_log['replaced']}\n"
            f"\tmax. new: {exploration_log['max_novel_representations']}\n"
            f"\tnew: {exploration_log['new']}\n"
            f"\ttotal: {exploration_log['total']}\n"
            f"\tmin. mass: {exploration_log['min_mass']}\n"
            f"\tmax. mass: {exploration_log['max_mass']}\n"
            f"\tavg. mass: {exploration_log['avg_mass']}\n"
        )
        return {
            "network": network_kwargs,
            "nn_training": nn_logs,
            "compression": compression_log,
            "exploration": exploration_log
        }

    def load_representation(self, r):
        with open(os.path.join(self.representations_path, r)) as f:
            return json.load(f)

    def load_dsl(self, dsl_name):
        with open(self.dsl_file(dsl_name)) as f:
            dsl = json.load(f)
            self.dsl_mass = dsl["mass"]
        self.dsl_name = dsl_name

    def dsl_file(self, dsl_name=None):
        if hasattr(self, "dsl_name"):
            if dsl_name is None:
                dsl_name = self.dsl_name
        elif dsl_name is None:
            raise ValueError(
                "attribute `dsl_name` does not exist and `dsl_name` parameter was not supplied")
        return os.path.join(self.dsl_path, dsl_name + ".json")

    @ property
    def n_representations(self):
        return len(self.representations)

    def explore(self,
                n_retained,
                next_dsl_name="dsl",
                *,
                exploration_timeout,
                program_size_limit,
                eval_timeout,
                eval_attempts,
                max_diff,
                max_conn):
        max_novel_representations = n_retained - self.n_representations
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        log = utilities.explore(self.name_of_domain,
                                self.representations,
                                max_novel_representations,
                                self.dsl_file(),
                                self.dsl_file(dsl_name=next_dsl_name),
                                self.representations_path,
                                exploration_timeout=exploration_timeout,
                                program_size_limit=program_size_limit,
                                eval_timeout=eval_timeout,
                                eval_attempts=eval_attempts,
                                max_diff=max_diff,
                                max_conn=max_conn)
        self.load_dsl(next_dsl_name)
        self._load_representations(dict(log["replacements"]))
        del log["replacements"]
        log.update(
            total=self.n_representations,
            n_retained=n_retained,
            max_novel_representations=max_novel_representations,
            prev_dsl_name=self.dsl_name,
            next_dsl_name=next_dsl_name,
            exploration_timeout=exploration_timeout,
            program_size_limit=program_size_limit,
            eval_timeout=eval_timeout,
            eval_attempts=eval_attempts,
            max_diff=max_diff,
            max_conn=max_conn,
            min_mass=self.min_mass,
            max_mass=self.max_mass,
            avg_mass=self.avg_mass)
        return log

    def compress(self,
                 next_dsl_name="dsl",
                 stitch_compression=True,
                 **kwargs):
        if stitch_compression:
            log = self._stitch_compress(
                next_dsl_name, **kwargs)
        else:
            log = self._dreamcoder_compress(
                next_dsl_name, **kwargs)
        log.update(stitch_compression=stitch_compression)
        return log

    def _dreamcoder_compress(self,
                             next_dsl_name,
                             *,
                             iterations,
                             beam_size,
                             n_beta_inversions,
                             n_invention_sizes,
                             n_exactly_scored,
                             primitive_size_penalty,
                             dsl_size_penalty,
                             invention_name_prefix,
                             verbosity=0):
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        result = utilities.dreamcoder_compress(self.representations,
                                               self.name_of_domain,
                                               self.dsl_file(),
                                               self.dsl_file(
                                                   dsl_name=next_dsl_name),
                                               self.representations_path,
                                               iterations=iterations,
                                               beam_size=beam_size,
                                               n_beta_inversions=n_beta_inversions,
                                               n_invention_sizes=n_invention_sizes,
                                               n_exactly_scored=n_exactly_scored,
                                               primitive_size_penalty=primitive_size_penalty,
                                               dsl_size_penalty=dsl_size_penalty,
                                               invention_name_prefix=invention_name_prefix,
                                               verbosity=verbosity)
        result.update(
            iterations=iterations,
            beam_size=beam_size,
            n_beta_inversions=n_beta_inversions,
            n_invention_sizes=n_invention_sizes,
            n_exactly_scored=n_exactly_scored,
            primitives_size_penalty=primitive_size_penalty,
            dsl_size_penalty=dsl_size_penalty,
            invention_name_prefix=invention_name_prefix)
        if result["n_added"] > 0:
            self.previous_abstractions += result["n_added"]
            result.update(
                prev_dsl_name=self.dsl_name,
                next_dsl_name=next_dsl_name)
            self._load_representations(dict(result["replacements"]))
            del result["replacements"]
            self.dsl_name = next_dsl_name
            self.dsl_mass = result["next_dsl_mass"]
        return result

    def _stitch_compress(self,
                         next_dsl_name,
                         *,
                         iterations,
                         n_beta_inversions,
                         threads,
                         verbose,
                         **stitch_kwargs):
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        stitch_programs = []
        for filename in self.representations:
            with open(os.path.join(self.representations_path, filename)) as f:
                contents = json.load(f)
                stitch_programs.append(contents["stitch_program"])
        res = utilities.stitch_compress(
            stitch_programs,
            self.previous_abstractions,
            iterations=iterations,
            n_beta_inversions=n_beta_inversions,
            threads=threads,
            verbose=verbose,
            **stitch_kwargs)
        result = {
            "iterations": iterations,
            "n_beta_inversions": n_beta_inversions
        }
        result.update(**stitch_kwargs)
        invented_primitives = [[name, body] for name, body in sorted(
            ((a.name, a.body) for a in res.abstractions), key=lambda a: int(a[0].split("_")[-1]))]
        result.update(n_added=len(invented_primitives))
        if result["n_added"] > 0:
            self.previous_abstractions += result["n_added"]
            replacements = [[prev, cur] for prev, cur in zip(
                res.json["original"], res.json["rewritten"]) if prev != cur]
            resp = utilities.incorporate_stitch(
                replacements, invented_primitives, self.name_of_domain, self.dsl_file(
                ), self.dsl_file(dsl_name=next_dsl_name),
                self.representations_path)
            self._load_representations(dict(resp["replacements"]))
            del resp["replacements"]
            result.update(resp)
            result.update(
                prev_dsl_name=self.dsl_name,
                next_dsl_name=next_dsl_name)
            self.dsl_name = next_dsl_name
            self.dsl_mass = resp["next_dsl_mass"]
        return result

    def _incremented_dsl_name(self, name):
        parts = self.dsl_name.split("_")
        prev_root, prev_number = "_".join(parts[:-1]), int(parts[-1])
        next_number = str(prev_number + 1)
        if name is None:
            return "_".join(("_".join(prev_root), next_number))
        else:
            return "_".join((name, next_number))

    def _load_representations(self, replacements):
        self.replacements = replacements
        self.representations = [(self.replacements[r] if r in self.replacements else r)
                                for r in self.representations]
        for i, discards in enumerate(self.discards):
            for j, d in enumerate(discards):
                if d in self.replacements:
                    self.discards[i][j] = self.replacements[d]
                    if d in os.listdir(self.discards_path):
                        os.remove(os.path.join(self.discards_path, d))
                        shutil.copy(os.path.join(self.representations_path,
                                                 self.replacements[d]),
                                    os.path.join(self.discards_path, self.replacements[d]))
        self.representations.extend(set(os.listdir(self.representations_path)) -
                                    set(self.representations))
        self.masses, self.programs = self._fetch_meta()
        self.min_mass = min(self.masses.values())
        self.max_mass = max(self.masses.values())
        self.avg_mass = sum(self.masses.values(), start=0) / len(self.masses)

    def _fetch_meta(self):
        masses, programs = {}, {}
        for filename in self.representations:
            with open(os.path.join(self.representations_path, filename)) as f:
                contents = json.load(f)
                masses[filename] = contents["mass"]
                programs[filename] = contents["program"]
        return masses, programs

    def mass_of_representations(self, representations):
        return sum(self.masses[r] for r in representations) / len(representations)

    def clear_representations(self):
        for filename in os.listdir(self.representations_path):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.representations_path, filename))
        self.representations = []
        self.masses, self.programs = self._fetch_meta()
        self.repr_usage = defaultdict(int)

    def clear_visualizations(self):
        for filename in os.listdir(self.visualizations_path):
            if filename.endswith(".svg"):
                os.remove(os.path.join(self.visualizations_path, filename))

    def clear_discards(self):
        for filename in os.listdir(self.discards_path):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.discards_path, filename))
        self.discards = []

    def clear_dsls(self, reset_dsl_name="dsl_0"):
        reset_dsl_file = f"{reset_dsl_name}.json"
        for dsl_file in os.listdir(self.dsl_path):
            if dsl_file != reset_dsl_file:
                os.remove(os.path.join(self.dsl_path, dsl_file))
        self.load_dsl(reset_dsl_name)

    def save_state(self, dest):
        shutil.copytree(self.dsl_path, os.path.join(dest, "dsls"))
        shutil.copytree(self.representations_path,
                        os.path.join(dest, "representations"))
        shutil.copytree(self.visualizations_path,
                        os.path.join(dest, "visualizations"))
        shutil.copytree(self.discards_path,
                        os.path.join(dest, "discards"))
        with open(os.path.join(dest, "discard_hist.json"), "w") as f:
            json.dump(self.discards, f)

    def visualize(self, to_visualize=None, dir=None):
        if dir is None:
            dir = self.representations_path
        if to_visualize is None:
            to_visualize = os.listdir(dir)
        else:
            to_visualize = list(set(to_visualize))
        visualized = 0
        for filename in to_visualize:
            if filename.endswith(".json"):
                with open(os.path.join(dir, filename)) as f:
                    contents = json.load(f)
                name = filename[:-5]
                graph = pgv.AGraph(labelloc="t", label=name)
                nodes, edges, n_elt = set(), set(), -1
                for [[node_1, _port], [edge, node_2]] in contents["output"]["edges"]:
                    edges.add((node_1, edge, node_2) if node_1 <
                              node_2 else (node_2, edge, node_1))
                    nodes.add(node_1)
                    nodes.add(node_2)
                    n_elt = max([node_1, edge, node_2])
                for node in nodes:
                    graph.add_node(
                        node, color=f"0.722 {node / n_elt:1.3f} 0.810")
                for (a, edge, b) in edges:
                    graph.add_edge((a, b), label=edge,
                                   color=f"0.722 {edge / n_elt:1.3f} 0.810",
                                   fontsize=12.0)
                graph.layout()
                graph.draw(
                    os.path.join(self.visualizations_path, name + ".svg"))
                visualized += 1
        print(f"produced {visualized} visualizations.")
