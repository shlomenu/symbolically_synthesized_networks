from typing import Optional, Tuple, List
import os
import json
from collections import defaultdict

import numpy as np
from einops import rearrange
import torch as th
from torch import nn
import torch.nn.functional as F

from restart_manager import RestartManager
import utilities


class Quantizer(nn.Module):

    def __init__(self,
                 name_of_domain,
                 dsl_name,
                 codebook_dim,
                 beta,
                 **explore_kwargs):
        super().__init__()
        self.name_of_domain = name_of_domain
        self.previous_abstractions = 0
        self.representations_path = os.path.join(
            self.name_of_domain, "representations")
        self.visualizations_path = os.path.join(
            self.name_of_domain, "visualizations")
        self.dsl_path = os.path.join(self.name_of_domain, "dsls")
        self.load_dsl(dsl_name)
        self.representations = sorted((filename for filename in os.listdir(
            self.representations_path) if filename.endswith(".json")))
        self.masses, self.programs = self._fetch_meta()
        self.repr_usage = defaultdict(int)
        self.explore_kwargs = explore_kwargs
        self.codebook_dim = codebook_dim
        self.in_codebook = self.initial_codebook()
        self.out_codebook = self.initial_codebook()
        self.in_restart_manager = RestartManager(len(self))
        self.out_restart_manager = RestartManager(len(self))
        self.in_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.out_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.beta = beta

    def load_dsl(self, dsl_name):
        with open(self.dsl_file(dsl_name)) as f:
            dsl = json.load(f)
            self.dsl_mass = dsl["mass"]
        self.dsl_name = dsl_name

    def initial_codebook(self):
        codebook = nn.Embedding(len(self), self.codebook_dim)
        codebook.weight.data.uniform_(-1. / (len(self) if len(self)
                                      > 0. else 1.), 1. / (len(self) if len(self) > 0. else 1.))
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def dsl_file(self, dsl_name=None):
        if hasattr(self, "dsl_name"):
            if dsl_name is None:
                dsl_name = self.dsl_name
        elif dsl_name is None:
            raise ValueError(
                "attribute `dsl_name` does not exist and `dsl_name` parameter was not supplied")
        return os.path.join(self.dsl_path, dsl_name + ".json")

    def __len__(self):
        return len(self.representations)

    def forward(self, x, quantization_noise_std=0.):
        in_shape = x.shape
        x = rearrange(x, "b s c -> b (s c)")
        assert x.size(1) == self.codebook_dim
        (x_quantized, ), (x_encoding_inds, ) = \
            self.nearest_neighbors(x, self.in_codebook,
                                   quantization_noise_std, noisy=False)
        recon, in_restarts, filenames = self.reconstruct(
            x_encoding_inds)  # type: ignore
        if in_restarts:
            self.in_restarts = (x.detach(), in_restarts)
        else:
            self.in_restarts = None
        (recon_quantized_det, recon_quantized_noisy), (_, recon_encoding_inds_noisy) = \
            self.nearest_neighbors(
                recon, self.out_codebook, quantization_noise_std)
        self.out_restarts, out_restarts = None, []
        for cmds in recon_encoding_inds_noisy.tolist():
            out_restarts: List[int] = self.out_restart_manager.find_restarts(
                cmds)
        if out_restarts:
            self.out_restarts = (x.detach(), out_restarts)
        out = x + (recon_quantized_noisy - x).detach()
        self.loss = \
            self.closeness_loss(x, recon_quantized_det, recon_quantized_noisy) + \
            self.closeness_loss(x, x_quantized) + \
            self.closeness_loss(recon, recon_quantized_det,
                                quantized_latents_noisy=recon_quantized_noisy)
        self.filenames = filenames
        return out.reshape(in_shape)

    def nearest_neighbors(self, flat_latents, codebook, noise_std, noisy=True):
        K = codebook.weight.size(0)

        dist = th.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            th.sum(codebook.weight ** 2, dim=1) - \
            2 * th.matmul(flat_latents,
                          codebook.weight.t())  # shape: b k

        encoding_inds_det = dist.argmin(
            dim=1).unsqueeze(dim=1)

        if noisy:
            encoding_inds = [encoding_inds_det, (encoding_inds_det + th.round(
                th.normal(mean=0,
                          std=noise_std,
                          size=encoding_inds_det.shape,
                          device=encoding_inds_det.device,
                          dtype=th.float)).long()).clamp(0, K - 1)]
        else:
            encoding_inds = [encoding_inds_det]

        quantized_latents = []
        for inds in encoding_inds:
            encoding_one_hot = th.zeros(inds.size(0) * inds.size(1),
                                        K,
                                        device=flat_latents.device)  # shape: b k
            encoding_one_hot.scatter_(dim=1, index=inds, value=1)
            quantized_latents.append(
                th.matmul(encoding_one_hot, codebook.weight))

        return quantized_latents, encoding_inds

    def closeness_loss(self, latents, quantized_latents_det, quantized_latents_noisy=None):
        adjusted_quantized_latents = quantized_latents_det if quantized_latents_noisy is None else quantized_latents_noisy
        return self.beta * F.mse_loss(latents, quantized_latents_det.detach()) + \
            F.mse_loss(adjusted_quantized_latents, latents.detach())

    def apply_restarts(self):
        if self.in_restarts is not None:
            starts, to_restart = self.in_restarts
            randomized = starts[np.random.choice(starts.size(0),
                                                 min(len(to_restart), starts.size(0)), replace=False)]
            self.in_codebook.weight.data[to_restart] = randomized
            self.in_restart_manager.reset_count(to_restart)
            self.in_restarts = None
        if self.out_restarts is not None:
            starts, to_restart = self.out_restarts
            randomized = starts[np.random.choice(starts.size(0),
                                                 min(len(to_restart), starts.size(0)), replace=False)]
            self.out_codebook.weight.data[to_restart] = randomized
            self.out_restart_manager.reset_count(to_restart)
            self.out_restarts = None

    def reconstruct(self, _selections):
        pass

    def make_frontier(self, representations_limit, repr_usage=None):
        log = {}
        if repr_usage is not None:
            self.repr_usage = repr_usage
        if len(self.repr_usage) > representations_limit:
            log.update(truncated_usages=True)
            total = sum(self.repr_usage.values())
            frontier = np.random.choice(
                list(self.repr_usage.keys()),
                int(representations_limit),
                replace=True,
                p=[count / total for count in self.repr_usage.values()]).tolist()
            log.update(frontier_div=len(
                self.repr_usage) / len(self))
            log.update(
                frontier_mass=self.mass_of_representations(frontier))
        else:
            log.update(truncated_usages=False)
            frontier = []
            for filename, count in self.repr_usage.items():
                frontier += [filename] * count
            log.update(frontier_div=len(self.repr_usage) / len(self))
            log.update(
                frontier_mass=self.mass_of_representations(frontier))
        return frontier, log

    def explore(self,
                frontier,
                exploration_timeout,
                next_dsl_name="dsl",
                *,
                eval_timeout,
                eval_attempts):
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        log = utilities.explore(self.name_of_domain,
                                frontier,
                                self.dsl_file(),
                                self.dsl_file(dsl_name=next_dsl_name),
                                self.representations_path,
                                exploration_timeout,
                                eval_timeout=eval_timeout,
                                eval_attempts=eval_attempts,
                                **self.explore_kwargs)
        self._load_representations(dict(log["replacements"]))
        del log["replacements"]
        log.update(
            total=len(self),
            exploration_timeout=exploration_timeout,
            min_mass=self.min_mass,
            max_mass=self.max_mass,
            avg_mass=self.avg_mass,
            prev_dsl_name=self.dsl_name,
            next_dsl_name=next_dsl_name,
            eval_timeout=eval_timeout,
            eval_attempts=eval_attempts)
        self.dsl_name = next_dsl_name
        if log["new"] > 0:
            in_codebook, out_codebook = self.initial_codebook(), self.initial_codebook()
            self.in_codebook = in_codebook.to(
                device=self.in_codebook.weight.device)
            self.out_codebook = out_codebook.to(
                device=self.out_codebook.weight.device)
            for code in range(len(self)):
                self.in_restart_manager.add_code(code)
                self.out_restart_manager.add_code(code)
            self.in_restarts, self.out_restarts = None, None
        return log

    def compress(self,
                 frontier,
                 next_dsl_name="dsl",
                 stitch_compression=True,
                 **kwargs):
        if stitch_compression:
            log = self._stitch_compress(
                frontier, next_dsl_name, **kwargs)
        else:
            log = self._dreamcoder_compress(
                frontier, next_dsl_name, **kwargs)
        log.update(stitch_compression=stitch_compression)
        return log

    def _dreamcoder_compress(self,
                             frontier,
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
        result = utilities.dreamcoder_compress(frontier,
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
                         frontier,
                         next_dsl_name,
                         *,
                         iterations,
                         n_beta_inversions,
                         threads,
                         verbose,
                         **stitch_kwargs):
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        frontier_programs = []
        for filename in frontier:
            with open(os.path.join(self.representations_path, filename)) as f:
                contents = json.load(f)
                frontier_programs.append(contents["stitch_program"])
        res = utilities.stitch_compress(
            frontier_programs,
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
            non_frontier = list(set(self.representations) - set(frontier))
            if non_frontier:
                non_frontier_programs = []
                for filename in non_frontier:
                    with open(os.path.join(self.representations_path, filename)) as f:
                        contents = json.load(f)
                        non_frontier_programs.append(
                            contents["stitch_program"])
                replacements.extend(([prev, cur] for prev, cur in zip(
                    non_frontier_programs, utilities.stitch_rewrite(
                        non_frontier_programs, res.abstractions))))
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
        self.representations = [(self.replacements[file] if file in self.replacements else file)
                                for file in self.representations]
        self.representations.extend(
            set(os.listdir(self.representations_path)) - set(self.representations))
        self.masses, self.programs = self._fetch_meta()
        self.min_mass = min(self.masses.values())
        self.max_mass = max(self.masses.values())
        self.avg_mass = sum(self.masses.values()) / len(self.masses)

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

    def visualize(self, to_visualize=None):
        pass
