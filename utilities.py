import json
import subprocess

import stitch_core

EXPLORE_BINARY_LOC = "program_synthesis/_build/default/explore.exe"
DREAMCODER_COMPRESS_BINARY_LOC = "program_synthesis/_build/default/dreamcoder_compress.exe"
INCORPORATE_STITCH_BINARY_LOC = "program_synthesis/_build/default/incorporate_stitch.exe"


def explore(domain,
            frontier,
            dsl_file,
            next_dsl_file,
            representations_dir,
            exploration_timeout,
            *,
            eval_timeout,
            eval_attempts,
            **kwargs):
    json_msg = {
        "domain": domain,
        "frontier": frontier,
        "dsl_file": dsl_file,
        "next_dsl_file": next_dsl_file,
        "representations_dir": representations_dir,
        "exploration_timeout": exploration_timeout,
        "eval_timeout": eval_timeout,
        "attempts": eval_attempts
    }
    json_msg.update(kwargs)
    return invoke_binary_with_json(EXPLORE_BINARY_LOC, json_msg)


def dreamcoder_compress(frontier,
                        domain,
                        dsl_file,
                        next_dsl_file,
                        representations_dir,
                        *,
                        iterations,
                        beam_size,
                        n_beta_inversions,
                        n_invention_sizes,
                        n_exactly_scored,
                        primitive_size_penalty,
                        dsl_size_penalty,
                        invention_name_prefix,
                        verbosity,
                        **kwargs):
    json_msg = {
        "frontier": frontier,
        "domain": domain,
        "dsl_file": dsl_file,
        "next_dsl_file": next_dsl_file,
        "representations_dir": representations_dir,
        "iterations": iterations,
        "beam_size": beam_size,
        "n_beta_inversions": n_beta_inversions,
        "n_invention_sizes": n_invention_sizes,
        "n_exactly_scored": n_exactly_scored,
        "primitive_size_penalty": primitive_size_penalty,
        "dsl_size_penalty": dsl_size_penalty,
        "invention_name_prefix": invention_name_prefix,
        "verbosity": verbosity
    }
    json_msg.update(kwargs)
    return invoke_binary_with_json(DREAMCODER_COMPRESS_BINARY_LOC, json_msg)


def stitch_compress(frontier,
                    previous_abstractions,
                    *,
                    iterations,
                    n_beta_inversions,
                    threads,
                    verbose,
                    **stitch_kwargs):
    stitch_kwargs.update(dynamic_batch=True)
    return stitch_core.compress(frontier, iterations, max_arity=n_beta_inversions, threads=threads,
                                silent=(not verbose), previous_abstractions=previous_abstractions,
                                **stitch_kwargs)


def stitch_rewrite(non_frontier,
                   abstractions,
                   **stitch_kwargs):
    stitch_kwargs.update(dynamic_batch=True)
    return stitch_core.rewrite(non_frontier, abstractions, **stitch_kwargs)


def incorporate_stitch(replacements,
                       invented_primitives,
                       domain,
                       dsl_file,
                       next_dsl_file,
                       representations_dir):
    json_msg = {
        "replacements": replacements,
        "invented_primitives": invented_primitives,
        "domain": domain,
        "dsl_file": dsl_file,
        "next_dsl_file": next_dsl_file,
        "representations_dir": representations_dir
    }
    return invoke_binary_with_json(INCORPORATE_STITCH_BINARY_LOC, json_msg)


def invoke_binary_with_json(path_to_binary, msg_json):
    msg = json.dumps(msg_json)
    try:
        process = subprocess.Popen(path_to_binary,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        resp, _ = process.communicate(bytes(msg, encoding="utf-8"))
    except OSError as exc:
        raise exc
    try:
        resp = json.loads(resp.decode("utf-8"))
    except Exception as e:
        eprint("Could not parse json.")
        with open("/tmp/_message", "w") as handle:
            handle.write(msg)
        with open("/tmp/_response", "w") as handle:
            handle.write(resp.decode("utf-8"))
        raise e
    return resp
