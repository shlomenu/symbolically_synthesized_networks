from typing import Optional, Tuple, List
import os
import json
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import utilities


class RestartManager:

    def __init__(self, K, idleness_limit=10):
        self.idleness_limit, self.K = idleness_limit, K
        self.utilization = {code: 0 for code in range(K)}

    def add_code(self, code: int):
        if code not in self.utilization:
            self.utilization[code] = 0
        self.K = len(self.utilization)

    def restrict(self, size):
        self.utilization = {code: count for code,
                            count in self.utilization.items() if code < size}

    def reset_count(self, selections):
        for code in selections:
            self.utilization[code] = 0

    def reset(self, size):
        self.restrict(size)
        self.reset_count(range(size))
        for code in range(size):
            self.add_code(code)

    def find_restarts(self, selections: List[int]):
        self.reset_count(selections)
        refreshed = set(selections)
        for code in range(self.K):
            if code not in refreshed:
                self.utilization[code] = self.utilization[code] + 1
        ranked = sorted(
            self.utilization.items(), key=(lambda x: x[1]), reverse=True)
        return [
            code for (code, since_used) in ranked
            if since_used > self.idleness_limit][:len(selections)]


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
        self.archetypes = deepcopy(self.representations)
        self.discards = []
        self.masses, self.programs = self._fetch_meta()
        self.clear_usages()
        self.explore_kwargs = explore_kwargs
        self.codebook_dim = codebook_dim
        self.base_codebook = self.initial_codebook()
        self.archetypal_codebook = self.initial_codebook()
        self.restart_manager = RestartManager(len(self))
        self.restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.beta = beta

    def load_dsl(self, dsl_name):
        with open(self.dsl_file(dsl_name)) as f:
            dsl = json.load(f)
            self.dsl_mass = dsl["mass"]
        self.dsl_name = dsl_name

    def _new_codebook(self, size, dim):
        codebook = nn.Embedding(size, dim)
        codebook.weight.data.uniform_(-1. / (size if size
                                      > 0. else 1.), 1. / (size if size > 0. else 1.))
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def initial_codebook(self):
        return self._new_codebook(len(self), self.codebook_dim)

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

    def finish_epoch(self, smoothing_factor=None, initial_unsmoothed=True):
        for r, usage in self.smoothed_usages.items():
            usages = self.complete_usages[r]
            if len(usages) == 1 and initial_unsmoothed:
                self.smoothed_usages[r] = usages[-1]
            else:
                assert smoothing_factor is not None and \
                    0. < smoothing_factor and smoothing_factor <= 1.
                self.smoothed_usages[r] = smoothing_factor * \
                    usages[-1] + (1. - smoothing_factor) * usage
            usages.append(0)

    def finish_training(self, n_archetypes, n_preserved):
        usage_order = [
            r for (r, _) in sorted(
                ((r, usage)
                    for r, usage in self.smoothed_usages.items()),
                key=lambda x: x[1], reverse=True)
        ]
        archetypes = deepcopy(usage_order[:n_archetypes])
        representations = deepcopy(usage_order[:n_preserved])
        discards = deepcopy(usage_order[n_preserved:])
        self.archetypal_codebook = self._transfer_codebook_subset(archetypes)
        self.base_codebook = self._transfer_codebook_subset(representations)
        self.archetypes, self.representations, self.discards = (
            archetypes, representations, discards)
        for r in self.discards:
            os.remove(os.path.join(self.representations_path, r))
        return len(self.discards)

    def _transfer_codebook_subset(self, rs):
        codebook = self._new_codebook(len(rs), self.codebook_dim).to(
            device=self.base_codebook.weight.device)
        codebook.weight.data[:len(rs)] = \
            self.base_codebook.weight.data[th.tensor(
                [self.representations.index(r) for r in rs])]
        return codebook

    def clear_usages(self):
        self.complete_usages = {r: [0] for r in self.representations}
        self.smoothed_usages = {r: 0 for r in self.representations}

    def forward(self, x):
        assert x.size(1) == self.codebook_dim
        x_codebook, x_encoding_inds = self.nearest_neighbors(x)
        x_quantized = x + (x_codebook - x).detach()
        for selection in x_encoding_inds.flatten().tolist():
            self.complete_usages[self.representations[selection]][-1] += 1
        out, restarts, self.filenames = self.structural_prediction(
            x_quantized, x_encoding_inds)  # type: ignore
        if restarts:
            self.restarts = (x.detach(), restarts)
        else:
            self.restarts = None
        if 0. <= self.beta and self.beta < 1.:
            self.loss = self.beta * F.mse_loss(x, x_codebook.detach()) + \
                F.mse_loss(x_codebook, x.detach())
        elif 1. <= self.beta:
            self.loss = F.mse_loss(x, x_codebook.detach()) + \
                (1. / self.beta) * F.mse_loss(x_codebook, x.detach())
        else:
            raise ValueError(f"beta cannot be negative: {self.beta}")
        return out

    def nearest_neighbors(self, latents):
        codebook = self.base_codebook if self.training else self.archetypal_codebook
        dist = th.sum(latents ** 2, dim=1, keepdim=True) + \
            th.sum(codebook.weight ** 2, dim=1) - \
            2 * th.matmul(latents, codebook.weight.t())  # shape: b k
        encoding_inds = dist.argmin(dim=1).unsqueeze(dim=1)
        encoding_one_hot = th.zeros(
            encoding_inds.size(0) * encoding_inds.size(1),
            codebook.weight.size(0), device=latents.device)  # shape: b k
        encoding_one_hot.scatter_(dim=1, index=encoding_inds, value=1)
        quantized_latents = th.matmul(
            encoding_one_hot, codebook.weight)
        return quantized_latents, encoding_inds

    def apply_restarts(self):
        if self.restarts is not None:
            starts, to_restart = self.restarts
            randomized = starts[np.random.choice(
                starts.size(0),
                min(len(to_restart), starts.size(0)),
                replace=False)]
            self.base_codebook.weight.data[to_restart] = randomized
            self.restart_manager.reset_count(to_restart)
            self.restarts = None

    def structural_prediction(self, _latents, _selections):
        pass

    def explore(self,
                n_retained,
                next_dsl_name="dsl",
                *,
                exploration_timeout,
                program_size_limit,
                eval_timeout,
                eval_attempts,
                max_diff):
        max_novel_representations = n_retained - len(self)
        next_dsl_name = self._incremented_dsl_name(next_dsl_name)
        log = utilities.explore(self.name_of_domain,
                                self.archetypes,
                                max_novel_representations,
                                self.dsl_file(),
                                self.dsl_file(dsl_name=next_dsl_name),
                                self.representations_path,
                                exploration_timeout=exploration_timeout,
                                program_size_limit=program_size_limit,
                                eval_timeout=eval_timeout,
                                eval_attempts=eval_attempts,
                                max_diff=max_diff,
                                **self.explore_kwargs)
        self._load_representations(dict(log["replacements"]))
        del log["replacements"]
        log.update(
            total=len(self),
            n_retained=n_retained,
            max_novel_representations=max_novel_representations,
            prev_dsl_name=self.dsl_name,
            next_dsl_name=next_dsl_name,
            exploration_timeout=exploration_timeout,
            program_size_limit=program_size_limit,
            eval_timeout=eval_timeout,
            eval_attempts=eval_attempts,
            max_diff=max_diff,
            min_mass=self.min_mass,
            max_mass=self.max_mass,
            avg_mass=self.avg_mass)
        self.dsl_name = next_dsl_name
        if log["new"] > 0:
            self.base_codebook = self.initial_codebook().to(
                device=self.base_codebook.weight.device)
            self.restart_manager.reset(len(self))
            self.restarts = None
            self.clear_usages()
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
        result = utilities.dreamcoder_compress(self.archetypes,
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
        for filename in self.archetypes:
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
            non_archetypal = list(
                set(self.representations) - set(self.archetypes))
            if non_archetypal:
                non_archetypal_programs = []
                for filename in non_archetypal:
                    with open(os.path.join(self.representations_path, filename)) as f:
                        contents = json.load(f)
                        non_archetypal_programs.append(
                            contents["stitch_program"])
                replacements.extend(([prev, cur] for prev, cur in zip(
                    non_archetypal_programs, utilities.stitch_rewrite(
                        non_archetypal_programs, res.abstractions))))
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
        self.archetypes = [(self.replacements[r] if r in self.replacements else r)
                           for r in self.archetypes]
        self.complete_usages = dict(
            (((self.replacements[r] if r in self.replacements else r), c) for r, c in self.complete_usages.items()))
        self.smoothed_usages = dict(
            (((self.replacements[r] if r in self.replacements else r), c)
             for r, c in self.smoothed_usages.items())
        )
        new = set(os.listdir(self.representations_path)) - \
            set(self.representations)
        self.representations.extend(new)
        self.complete_usages.update(((r, [0]) for r in new))
        self.smoothed_usages.update(((r, 0) for r in new))
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

    def visualize(self, to_visualize=None):
        pass


class VQBaseline(nn.Module):

    def __init__(self,
                 codebook_size,
                 codebook_dim,
                 beta):
        super().__init__()
        self.filenames = None
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = self.initial_codebook()
        self.restart_manager = RestartManager(len(self))
        self.restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.beta = beta

    def __len__(self):
        return self.codebook_size

    def initial_codebook(self):
        codebook = nn.Embedding(len(self), self.codebook_dim)
        codebook.weight.data.uniform_(-1. / (len(self) if len(self)
                                      > 0. else 1.), 1. / (len(self) if len(self) > 0. else 1.))
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def forward(self, x):
        assert x.size(1) == self.codebook_dim
        x_codebook, x_encoding_inds = self.nearest_neighbors(x)
        out = x + (x_codebook - x).detach()
        restarts = self.restart_manager.find_restarts(
            x_encoding_inds.flatten().tolist())
        if restarts:
            self.restarts = (x.detach(), restarts)
        else:
            self.restarts = None
        if 0. <= self.beta and self.beta < 1.:
            self.loss = self.beta * F.mse_loss(x, x_codebook.detach()) + \
                F.mse_loss(x_codebook, x.detach())
        elif 1. <= self.beta:
            self.loss = F.mse_loss(x, x_codebook.detach()) + \
                (1. / self.beta) * F.mse_loss(x_codebook, x.detach())
        else:
            raise ValueError(f"beta cannot be negative: {self.beta}")
        return out

    def nearest_neighbors(self, latents):
        K = self.codebook.weight.size(0)
        dist = th.sum(latents ** 2, dim=1, keepdim=True) + \
            th.sum(self.codebook.weight ** 2, dim=1) - \
            2 * th.matmul(latents,
                          self.codebook.weight.t())  # shape: b k
        encoding_inds = dist.argmin(dim=1).unsqueeze(dim=1)
        encoding_one_hot = th.zeros(
            encoding_inds.size(0) * encoding_inds.size(1),
            K, device=latents.device)  # shape: b k
        encoding_one_hot.scatter_(dim=1, index=encoding_inds, value=1)
        quantized_latents = th.matmul(
            encoding_one_hot, self.codebook.weight)
        return quantized_latents, encoding_inds

    def apply_restarts(self):
        if self.restarts is not None:
            starts, to_restart = self.restarts
            randomized = starts[np.random.choice(starts.size(0),
                                                 min(len(to_restart), starts.size(0)), replace=False)]
            self.codebook.weight.data[to_restart] = randomized
            self.restart_manager.reset_count(to_restart)
            self.restarts = None

    def finish_epoch(self):
        pass

    def finish_training(self):
        pass
