import sys
import json
import subprocess
import os
import psutil
import time

import torch as th
from torchvision.utils import save_image

DEDUPLICATE_BINARY_LOC = "program_synthesis/_build/default/deduplicate.exe"
EXECUTE_BINARY_LOC = "program_synthesis/_build/default/execute.exe"
COMPRESS_BINARY_LOC = "program_synthesis/_build/default/compress.exe"


def deduplicate(domain, executed_programs_dir):
    return invoke_binary_with_json(DEDUPLICATE_BINARY_LOC, {
        "executed_programs_dir": executed_programs_dir,
        "domain": domain
    })


def compress(domain,
             dsl_file,
             next_dsl_file,
             executed_programs_dir,
             iterations=1,
             beam_size=3,
             top_i=3,
             dsl_size_penalty=0.,
             primitive_size_penalty=1.,
             n_beta_inversions=3,
             n_cores=1,
             verbosity=0):
    json_msg = {
        "domain": domain,
        "dsl_file": dsl_file,
        "next_dsl_file": next_dsl_file,
        "executed_programs_dir": executed_programs_dir,
        "iterations": iterations,
        "beam_size": beam_size,
        "top_i": top_i,
        "dsl_size_penalty": dsl_size_penalty,
        "primitive_size_penalty": primitive_size_penalty,
        "n_beta_inversions": n_beta_inversions,
        "n_cores": n_cores,
        "verbosity": verbosity
    }
    return invoke_binary_with_json(COMPRESS_BINARY_LOC, json_msg)


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


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    flush_all()


def save_reconstruction(filename, autoencoder_psn, dataset, i,
                        quantization_noise_std, mode, device):
    assert (mode in ("nearest", "learned"))
    img, _ = dataset[i]
    img = th.unsqueeze(2 * (img / 255) - 1., 0).to(device)
    out, _, _ = autoencoder_psn(img, img, quantization_noise_std, mode)
    reconstructed_img = .5 * (out + 1.)
    save_image(reconstructed_img, filename)
    return reconstructed_img


# x11 color scheme
COLORS = [
    "red",
    "green",
    "blue",
    "cyan",
    "brown",
    "gold",
    "indigo",
    "lemonchiffon",
    "magenta",
    "midnightblue",
    "slategrey",
    "yellow",
]


class Timing(object):

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, _type, _value, _trackeback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            assert False, "Timing message should be string function"
        eprint("%s in %.1f seconds" % (message, dt))


def flush_all():
    sys.stdout.flush()
    sys.stderr.flush()


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def memory_usage_frac():
    return psutil.virtual_memory().percent


def memory_usage_gb():
    return psutil.virtual_memory().total / 10**9
