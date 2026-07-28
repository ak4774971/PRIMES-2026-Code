#!/usr/bin/env python3
"""
word2vect_experiments.py
------------------------
Standalone CBOW word2vec optimizer comparison (from word2vect.ipynb).

Configure parameters via the globals below, then run:
    python word2vect_experiments.py

Designed for remote runs (e.g. under GNU screen). On completion writes:
    experiment_runs/<timestamp>/report.md      # human report with embedded plots
    experiment_runs/<timestamp>/results.json   # machine-readable stats
    experiment_runs/<timestamp>/plots/*.png

Optional deps: cupy (GPU), psutil (nicer CPU/RAM sampling), matplotlib, tqdm.
"""

from __future__ import annotations

# =============================================================================
# Experiment configuration (edit these)
# =============================================================================

BATCH_SIZE = 2048 # default: 256
EMBED_DIM = 100
EPOCHS = 20
VOCAB_SIZE = 30000 
CONTEXT_WINDOW_SIZE = 3  
TEXT_CHARS = 168085_788 # default:5_000_000  max value: 168_085_788
SEED = 1000

OPTIMIZERS = ["sgd", "red", "redm", "adahessian", "adam"]
LR = {"sgd": 3.0, "default": 0.01}
LR_DECAY = {"sgd": 1.0, "default": 1.0}

RUN_BENCHMARKS = True
RUN_SPOT_CHECKS = True
SAMPLE_INTERVAL_SEC = 2.0

CORPUS_PATH = "AllCombined.txt"
SIMLEX_PATH = "SimLex-999/SimLex-999.txt"
WORDSIM_PATH = "wordsim353crowd.csv"
GOOGLE_PATH = "questions-words.txt"
OUTPUT_DIR = "experiment_runs"

GOOGLE_TOP_K = [3, 5]
SPOT_SIMILAR_WORDS = ["king", "apple", "war", "queen", "battle"]
SPOT_ANALOGIES = [
    ("man", "king", "woman"),
    ("son", "father", "daughter"),
]

DISPLAY_NAME = {
    "sgd": "SGD",
    "red": "RED",
    "redm": "RED-M",
    "adahessian": "AdaHessian",
    "adam": "Adam",
}

# =============================================================================
# Imports
# =============================================================================

import copy
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as ncpu

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **kwargs):
        return iterable


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Neural as Neur

np = Neur.np  # CuPy if available inside Neural.py, else NumPy

try:
    import psutil
except ImportError:
    psutil = None


# =============================================================================
# Resource sampling
# =============================================================================


def _read_rss_bytes() -> Optional[int]:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # kB
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _read_cpu_percent(prev: Optional[Tuple[float, float]]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Return (cpu_percent, new_prev). prev is (proc_time, wall_time)."""
    if psutil is not None:
        try:
            # Non-blocking after first call; first call returns 0.0
            return float(psutil.Process(os.getpid()).cpu_percent(interval=None)), prev
        except Exception:
            pass
    try:
        with open("/proc/self/stat") as f:
            parts = f.read().split()
        # utime + stime in clock ticks
        utime = int(parts[13])
        stime = int(parts[14])
        proc_time = (utime + stime) / os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        wall = time.time()
        if prev is None:
            return None, (proc_time, wall)
        d_proc = proc_time - prev[0]
        d_wall = wall - prev[1]
        if d_wall <= 0:
            return None, (proc_time, wall)
        return 100.0 * d_proc / d_wall, (proc_time, wall)
    except Exception:
        return None, prev


def _read_vram_bytes() -> Tuple[Optional[int], Optional[int]]:
    """Return (used_bytes, total_bytes) or (None, None)."""
    # Prefer CuPy device mem info when available
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        used = int(total - free)
        return used, int(total)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0]
        used_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        return used_mb * 1024 * 1024, total_mb * 1024 * 1024
    except Exception:
        return None, None


def _summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "mean": None, "max": None, "n": 0}
    return {
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "max": float(max(values)),
        "n": len(values),
    }


class ResourceSampler:
    """Background thread that periodically samples CPU / RAM / VRAM."""

    def __init__(self, interval_sec: float = 2.0):
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._label = "global"
        self.samples: List[Dict[str, Any]] = []
        self._cpu_prev: Optional[Tuple[float, float]] = None
        if psutil is not None:
            try:
                psutil.Process(os.getpid()).cpu_percent(interval=None)
            except Exception:
                pass

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ResourceSampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            cpu, self._cpu_prev = _read_cpu_percent(self._cpu_prev)
            rss = _read_rss_bytes()
            vram_used, vram_total = _read_vram_bytes()
            with self._lock:
                label = self._label
            self.samples.append(
                {
                    "t": time.time(),
                    "label": label,
                    "cpu_percent": cpu,
                    "rss_bytes": rss,
                    "vram_used_bytes": vram_used,
                    "vram_total_bytes": vram_total,
                }
            )
            self._stop.wait(self.interval_sec)

    def summarize(self, label: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            rows = list(self.samples)
        if label is not None:
            rows = [r for r in rows if r["label"] == label]
        cpu = [r["cpu_percent"] for r in rows if r["cpu_percent"] is not None]
        rss = [r["rss_bytes"] for r in rows if r["rss_bytes"] is not None]
        vram = [r["vram_used_bytes"] for r in rows if r["vram_used_bytes"] is not None]
        vram_tot = [r["vram_total_bytes"] for r in rows if r["vram_total_bytes"] is not None]
        return {
            "cpu_percent": _summarize(cpu),
            "rss_bytes": _summarize(rss),
            "vram_used_bytes": _summarize(vram),
            "vram_total_bytes": max(vram_tot) if vram_tot else None,
            "n_samples": len(rows),
        }


def _fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _fmt_stats_bytes(s: Dict[str, Any]) -> str:
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"min={_fmt_bytes(s['min'])}, mean={_fmt_bytes(s['mean'])}, max={_fmt_bytes(s['max'])} (n={s['n']})"


def _fmt_stats_num(s: Dict[str, Any], unit: str = "") -> str:
    if not s or s.get("n", 0) == 0:
        return "n/a"
    u = f" {unit}" if unit else ""
    return f"min={s['min']:.1f}{u}, mean={s['mean']:.1f}{u}, max={s['max']:.1f}{u} (n={s['n']})"


# =============================================================================
# Data preparation
# =============================================================================

STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now",
]


def load_text(path: str, n_chars: int) -> str:
    valid = "abcdefghijklmnopqrstuvwxyz "
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Required corpus not found: {path}\n"
            f"Run download_files.py first or set CORPUS_PATH."
        )
    with open(p) as f:
        raw = f.read()
    low = raw.lower().replace("\n", " ")
    letters_only = "".join(ch for ch in low if ch in valid)
    return letters_only[:n_chars]


def build_dataset(txt: str, vocab_sz: int, window: int, seed: int):
    lst = [w for w in txt.split() if w != ""]
    filtered = [w for w in lst if w not in STOP_WORDS]
    vocab = [w for w, _ in Counter(filtered).most_common(vocab_sz) if w not in STOP_WORDS]
    vocab_sz_eff = len(vocab)
    vocab_st = set(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    data = [word2idx[w] for w in filtered if w in vocab_st]

    ncpu.random.seed(seed)
    x, y = [], []
    for i in range(window, len(data) - window):
        context = [data[j] for j in range(i - window, i + window + 1) if j != i]
        x.append(context)
        y.append(data[i])

    x = ncpu.array(x)
    y = ncpu.array(y)
    indices = ncpu.arange(len(x))
    ncpu.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    end = int(len(x_shuffled) * 0.9)
    return {
        "vocab": vocab,
        "vocab_st": vocab_st,
        "word2idx": word2idx,
        "vocab_sz": vocab_sz_eff,
        "X_train": x_shuffled[:end],
        "Y_train": y_shuffled[:end],
        "X_test": x_shuffled[end:],
        "Y_test": y_shuffled[end:],
        "n_tokens": len(data),
        "n_train": end,
        "n_test": len(x_shuffled) - end,
    }


def get_cbow(context, vocab_sz: int, window: int):
    mat = ncpu.zeros((len(context), vocab_sz))
    for idx, row in enumerate(context):
        for word in row:
            mat[idx, word] += 1.0 / (2 * window)
    return np.asarray(mat)


def get_one_hot(indices, vocab_sz: int):
    mat = np.zeros((len(indices), vocab_sz))
    mat[np.arange(len(indices)), np.asarray(indices)] = 1
    return mat


# =============================================================================
# Optimizers / training (ported from notebook)
# =============================================================================


def _as_float(x) -> float:
    return float(x)


def clone_network(N):
    N2 = copy.deepcopy(N)
    N2.set_params(np.array(N.get_params(), copy=True))
    return N2


def step_sgd(N_a, state, g):
    params = N_a.get_params()
    N_a.set_params(params - state["lr"] * g)
    return state


def step_red(N_a, state, g, lv, X):
    beta3, ell, eps = 0.9, 1.0, 1e-4
    d = g
    nrm2 = np.dot(d, d)
    p = N_a.get_params()
    N_a.set_params(p + eps * d)
    lp = N_a.forward(X.T)
    N_a.set_params(p - eps * d)
    lm = N_a.forward(X.T)
    N_a.set_params(p)
    ck = max(abs(_as_float((lp + lm - 2 * lv) / (eps**2)) / _as_float(nrm2)), 1e-20)
    state["chat"] = beta3 * state["chat"] + (1 - beta3) * ck
    ctilde = state["chat"] / (1 - beta3 ** (state["k"] + 1))
    Lk = max(ctilde, ck)
    rk = np.dot(d, g) / (2 * nrm2 * Lk)
    N_a.set_params(p - ell * rk * d)
    state["k"] += 1
    return state


def step_adam(N_a, state, g):
    b1, b2, e = 0.9, 0.999, 1e-8
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g**2
    mh = state["m"] / (1 - b1 ** (state["k"] + 1))
    vh = state["v"] / (1 - b2 ** (state["k"] + 1))
    p = N_a.get_params()
    p -= state["lr"] * mh / (np.sqrt(vh) + e)
    N_a.set_params(p)
    state["k"] += 1
    return state


def step_adahessian(N_a, state, g, loss_layer, X, Y):
    b1, b2, e, eps = 0.9, 0.999, 1e-4, 1e-4
    p = N_a.get_params()
    z = np.random.choice([-1.0, 1.0], size=len(p))
    N_a.set_params(p + eps * z)
    loss_layer.save_D = Y.T
    N_a.forward(X.T)
    gp, _ = N_a.backward(None)
    N_a.set_params(p)
    D = z * (gp - g) / eps
    D = np.clip(D, -2.0, 2.0)
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * (D**2)
    mh = state["m"] / (1 - b1 ** (state["k"] + 1))
    vh = state["v"] / (1 - b2 ** (state["k"] + 1))
    p -= state["lr"] * mh / (np.sqrt(vh) + e)
    N_a.set_params(p)
    state["k"] += 1
    return state


def step_redm(N_a, state, g, lv, X):
    beta3, b1, ell, eps = 0.9, 0.9, 1.0, 1e-4
    state["m"] = b1 * state["m"] + (1 - b1) * g
    mh = state["m"] / (1 - b1 ** (state["k"] + 1))
    d = mh
    nrm2 = np.dot(d, d)
    p = N_a.get_params()
    N_a.set_params(p + eps * d)
    lp = N_a.forward(X.T)
    N_a.set_params(p - eps * d)
    lm = N_a.forward(X.T)
    N_a.set_params(p)
    ck = max(abs(_as_float((lp + lm - 2 * lv) / (eps**2)) / _as_float(nrm2)), 1e-20)
    state["chat"] = beta3 * state["chat"] + (1 - beta3) * ck
    ctilde = state["chat"] / (1 - beta3 ** (state["k"] + 1))
    Lk = max(ctilde, ck)
    rk = 1 / (2 * Lk)
    p -= ell * rk * d
    N_a.set_params(p)
    state["k"] += 1
    return state


def build_net(
    optimizer: str,
    dim: int,
    epochs: int,
    batch: int,
    lr: float,
    decay: float,
    vocab_sz: int,
    window: int,
    X_train,
    Y_train,
):
    np.random.seed(SEED)
    history = []
    history_00 = []
    Nh = []

    N = Neur.Network([Neur.Dense(vocab_sz, dim), Neur.Dense(dim, vocab_sz)])
    loss_layer = Neur.Ilogit_and_KL(None)
    N_a = Neur.Network([N, loss_layer])

    state = {"k": 0, "lr": lr, "decay": decay}
    if optimizer in ["adahessian", "adam", "redm", "spadahessian", "spadam", "spredm"]:
        state["m"] = np.zeros(N_a.nb_params)
    if optimizer in ["adahessian", "adam", "spadahessian", "spadam"]:
        state["v"] = np.zeros(N_a.nb_params)
    if optimizer in ["red", "redm", "spredm"]:
        state["chat"] = 0.0

    start_time = time.time()
    times = []
    sz = len(Y_train)

    for epoch in range(epochs):
        total_loss = 0.0
        indices = ncpu.arange(sz)
        ncpu.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        batch_loop = tqdm(
            range(0, len(X_shuffled), batch),
            desc=f"{optimizer} epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        batches = 0
        for i in batch_loop:
            X_cur = get_cbow(X_shuffled[i : i + batch], vocab_sz, window)
            Y_cur = get_one_hot(Y_shuffled[i : i + batch], vocab_sz)
            loss_layer.save_D = Y_cur.T
            loss = N_a.forward(X_cur.T)
            total_loss += _as_float(loss)
            grads, _ = N_a.backward(None)
            history_00.append(_as_float(grads[0]))

            if optimizer == "sgd":
                state = step_sgd(N_a, state, grads)
            elif optimizer == "red":
                state = step_red(N_a, state, grads, loss, X_cur)
            elif optimizer == "redm":
                state = step_redm(N_a, state, grads, loss, X_cur)
            elif optimizer == "adam":
                state = step_adam(N_a, state, grads)
            elif optimizer == "adahessian":
                state = step_adahessian(N_a, state, grads, loss_layer, X_cur, Y_cur)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

            batches += 1
            if i % 50 == 0:
                try:
                    batch_loop.set_postfix({"loss": f"{total_loss / batches:.2f}"})
                except Exception:
                    pass

        history.append(total_loss / max(batches, 1))
        state["lr"] *= state["decay"]
        times.append(time.time() - start_time)
        Nh.append(clone_network(N))
        print(f"  [{optimizer}] epoch {epoch + 1}: train_loss={history[-1]:.4f}  t={times[-1]:.1f}s")

    return Nh, history, history_00, times


# =============================================================================
# Evaluation / benchmarks
# =============================================================================


def eval_test_loss(network, X_test, Y_test, vocab_sz: int, window: int, batch: int) -> float:
    loss_layer = Neur.Ilogit_and_KL(None)
    N_a = Neur.Network([network, loss_layer])
    total_loss = 0.0
    batches = 0
    for k in range(0, len(X_test), batch):
        X_cur = get_cbow(X_test[k : k + batch], vocab_sz, window)
        Y_cur = get_one_hot(Y_test[k : k + batch], vocab_sz)
        loss_layer.save_D = Y_cur.T
        loss = N_a.forward(X_cur.T)
        total_loss += float(loss)
        batches += 1
    return total_loss / max(batches, 1)


def normalized_embedding(network):
    embed = network.list_layers[0].A.T
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    return embed / norms


def spearman_rho(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    # Ensure backend arrays for argsort
    aa = np.asarray(aa)
    bb = np.asarray(bb)
    d_sq = (np.argsort(np.argsort(aa)) - np.argsort(np.argsort(bb))) ** 2
    return float(1 - (6 * np.sum(d_sq)) / (n * (n**2 - 1)))


def run_simlex_benchmark(filepath: str, norm_embed, vocab_st, word2idx) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.is_file():
        return {"skipped": True, "reason": f"file not found: {filepath}"}
    simlex, model = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().lower().split("\t")
            if len(parts) >= 4 and parts[0] in vocab_st and parts[1] in vocab_st:
                simlex.append(float(parts[3]))
                cos_sim = np.dot(norm_embed[word2idx[parts[0]]], norm_embed[word2idx[parts[1]]])
                model.append(float(cos_sim))
    rho = spearman_rho(simlex, model)
    return {"skipped": False, "n": len(simlex), "spearman_rho": rho}


def run_wordsim_benchmark(filepath: str, norm_embed, vocab_st, word2idx) -> Dict[str, Any]:
    path = Path(filepath)
    if not path.is_file():
        return {"skipped": True, "reason": f"file not found: {filepath}"}
    wordsim, model = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().lower().split(",")
            if len(parts) >= 3 and parts[0] in vocab_st and parts[1] in vocab_st:
                wordsim.append(float(parts[2]))
                cos_sim = np.dot(norm_embed[word2idx[parts[0]]], norm_embed[word2idx[parts[1]]])
                model.append(float(cos_sim))
    rho = spearman_rho(wordsim, model)
    return {"skipped": False, "n": len(wordsim), "spearman_rho": rho}


def run_google_benchmark(path: str, norm_embed, vocab, vocab_st, word2idx, k: int = 5) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"skipped": True, "reason": f"file not found: {path}"}
    tests = []
    with open(p) as f:
        for line in f:
            line = line.strip().lower()
            words = line.split()
            if len(words) == 4 and all(w in vocab_st for w in words):
                tests.append(tuple(words))
    valid = len(tests)
    if valid == 0:
        return {
            "skipped": False,
            "n": 0,
            "top1_acc": 0.0,
            f"top{k}_acc": 0.0,
            "top1": 0,
            f"top{k}": 0,
            "k": k,
        }
    top1 = 0
    topk = 0
    for a, b, c, ans in tests:
        vec_a = norm_embed[word2idx[a]]
        vec_b = norm_embed[word2idx[b]]
        vec_c = norm_embed[word2idx[c]]
        target = vec_b - vec_a + vec_c
        target = target / np.linalg.norm(target)
        similar = norm_embed @ target
        similar[word2idx[a]] = -100
        similar[word2idx[b]] = -100
        similar[word2idx[c]] = -100
        nearest = np.argsort(similar)[::-1]
        best_words = [vocab[int(idx)] for idx in nearest[:k]]
        if best_words[0] == ans:
            top1 += 1
        if ans in best_words:
            topk += 1
    return {
        "skipped": False,
        "n": valid,
        "top1": top1,
        "top1_acc": 100.0 * top1 / valid,
        f"top{k}": topk,
        f"top{k}_acc": 100.0 * topk / valid,
        "k": k,
    }


def get_similar(word: str, norm_embed, vocab, word2idx, top_k: int = 5) -> Dict[str, Any]:
    if word not in word2idx:
        return {"word": word, "error": "not in vocab"}
    word_vec = norm_embed[word2idx[word]]
    similar = norm_embed @ word_vec
    nearest = np.argsort(similar)[::-1]
    neighbors = []
    for i in range(1, top_k + 1):
        idx = int(nearest[i])
        neighbors.append({"word": vocab[idx], "score": float(similar[idx])})
    return {"word": word, "neighbors": neighbors}


def find_analogy(a: str, b: str, c: str, norm_embed, vocab, word2idx, top_k: int = 3) -> Dict[str, Any]:
    if any(w not in word2idx for w in (a, b, c)):
        return {"query": [a, b, c], "error": "one or more words missing from vocab"}
    target = norm_embed[word2idx[b]] - norm_embed[word2idx[a]] + norm_embed[word2idx[c]]
    target = target / np.linalg.norm(target)
    similar = norm_embed @ target
    nearest = np.argsort(similar)[::-1]
    results = []
    for idx in nearest:
        word = vocab[int(idx)]
        if word not in (a, b, c):
            results.append({"word": word, "score": float(similar[idx])})
        if len(results) == top_k:
            break
    return {"query": [a, b, c], "results": results}


# =============================================================================
# Plots & reporting
# =============================================================================


def save_plots(run_dir: Path, loss_history, runtime, test_losses, grad_history, optimizers: List[str]) -> List[str]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Train loss vs epoch
    plt.figure(figsize=(8, 5))
    for opt in optimizers:
        epochs = range(1, len(loss_history[opt]) + 1)
        label = f"{DISPLAY_NAME.get(opt, opt)} ({round(runtime[opt][-1])}s)"
        plt.plot(epochs, loss_history[opt], marker="o", markersize=3, label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Across Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = plots_dir / "train_loss_vs_epoch.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    saved.append("plots/train_loss_vs_epoch.png")

    # Train loss vs time
    plt.figure(figsize=(8, 5))
    for opt in optimizers:
        plt.plot(runtime[opt], loss_history[opt], marker="o", markersize=3, label=DISPLAY_NAME.get(opt, opt))
    plt.xlabel("Cumulative Time (seconds)")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = plots_dir / "train_loss_vs_time.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    saved.append("plots/train_loss_vs_time.png")

    # Test loss vs epoch
    if test_losses:
        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            epochs = range(1, len(test_losses[opt]) + 1)
            plt.plot(epochs, test_losses[opt], marker="o", markersize=3, label=DISPLAY_NAME.get(opt, opt))
        plt.xlabel("Epochs")
        plt.ylabel("Test Loss")
        plt.title("Test Loss Across Epochs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = plots_dir / "test_loss_vs_epoch.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        saved.append("plots/test_loss_vs_epoch.png")

    # Gradient of W[0][0]
    if grad_history:
        plt.figure(figsize=(8, 5))
        for opt in optimizers:
            ys = grad_history[opt]
            xs = ncpu.linspace(0, EPOCHS, len(ys))
            plt.plot(xs, ys, alpha=0.25, label=DISPLAY_NAME.get(opt, opt))
        plt.xlabel("Epoch")
        plt.ylabel("Gradient of W[0][0]")
        plt.title("Gradient Trace")
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = plots_dir / "grad_w00.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        saved.append("plots/grad_w00.png")

    return saved


def backend_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "array_module": getattr(np, "__name__", str(type(np))),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        info["gpu"] = name
        free, total = cp.cuda.runtime.memGetInfo()
        info["vram_total_bytes"] = int(total)
        info["vram_free_at_start_bytes"] = int(free)
        info["cupy"] = True
    except Exception as e:
        info["gpu"] = None
        info["cupy"] = False
        info["cupy_error"] = str(e)
    return info


def write_report_md(run_dir: Path, payload: Dict[str, Any], plot_paths: List[str]) -> None:
    cfg = payload["config"]
    hw = payload["hardware"]
    lines: List[str] = []
    lines.append(f"# Word2Vec Optimizer Experiments — {payload['run_id']}")
    lines.append("")
    lines.append(f"Generated: `{payload['finished_at']}`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    for k, v in cfg.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Hardware / backend")
    lines.append("")
    lines.append(f"- Host: `{hw.get('hostname')}`")
    lines.append(f"- Platform: `{hw.get('platform')}`")
    lines.append(f"- Python: `{hw.get('python')}`")
    lines.append(f"- Array backend: `{hw.get('array_module')}`")
    lines.append(f"- CPU count: `{hw.get('cpu_count')}`")
    lines.append(f"- GPU: `{hw.get('gpu')}`")
    lines.append(f"- CuPy: `{hw.get('cupy')}`")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    ds = payload["dataset"]
    lines.append(f"- Vocab size: **{ds['vocab_sz']}**")
    lines.append(f"- Tokens (filtered): **{ds['n_tokens']}**")
    lines.append(f"- Train pairs: **{ds['n_train']}**, test pairs: **{ds['n_test']}**")
    lines.append("")
    lines.append("## Training results")
    lines.append("")
    lines.append("| Optimizer | Final train loss | Final test loss | Wall time (s) |")
    lines.append("|---|---:|---:|---:|")
    for opt in cfg["OPTIMIZERS"]:
        r = payload["results"][opt]
        lines.append(
            f"| {DISPLAY_NAME.get(opt, opt)} | {r['train_loss'][-1]:.4f} | "
            f"{r['test_loss'][-1]:.4f} | {r['runtime_sec'][-1]:.1f} |"
        )
    lines.append("")
    lines.append(f"Total wall time (all optimizers + eval): **{payload['total_wall_sec']:.1f}s**")
    lines.append("")

    lines.append("## Resource usage")
    lines.append("")
    lines.append("Samples taken every "
                 f"`{cfg['SAMPLE_INTERVAL_SEC']}`s during the run.")
    lines.append("")
    lines.append("### Overall")
    ov = payload["resources"]["overall"]
    lines.append(f"- CPU%: {_fmt_stats_num(ov['cpu_percent'], '%')}")
    lines.append(f"- RSS: {_fmt_stats_bytes(ov['rss_bytes'])}")
    lines.append(f"- VRAM used: {_fmt_stats_bytes(ov['vram_used_bytes'])}")
    if ov.get("vram_total_bytes"):
        lines.append(f"- VRAM total: {_fmt_bytes(ov['vram_total_bytes'])}")
    lines.append("")
    lines.append("### Per optimizer")
    lines.append("")
    lines.append("| Optimizer | CPU% (mean) | RSS max | VRAM max |")
    lines.append("|---|---:|---:|---:|")
    for opt in cfg["OPTIMIZERS"]:
        s = payload["resources"]["per_optimizer"].get(opt, {})
        cpu = s.get("cpu_percent", {})
        rss = s.get("rss_bytes", {})
        vram = s.get("vram_used_bytes", {})
        cpu_m = f"{cpu['mean']:.1f}" if cpu and cpu.get("mean") is not None else "n/a"
        rss_m = _fmt_bytes(rss.get("max")) if rss else "n/a"
        vram_m = _fmt_bytes(vram.get("max")) if vram else "n/a"
        lines.append(f"| {DISPLAY_NAME.get(opt, opt)} | {cpu_m} | {rss_m} | {vram_m} |")
    lines.append("")

    if payload.get("benchmarks"):
        lines.append("## Benchmarks")
        lines.append("")
        for opt, b in payload["benchmarks"].items():
            lines.append(f"### {DISPLAY_NAME.get(opt, opt)}")
            lines.append("")
            sx = b.get("simlex", {})
            if sx.get("skipped"):
                lines.append(f"- SimLex-999: skipped ({sx.get('reason')})")
            else:
                lines.append(f"- SimLex-999: n={sx.get('n')}, Spearman ρ={sx.get('spearman_rho'):.4f}")
            ws = b.get("wordsim", {})
            if ws.get("skipped"):
                lines.append(f"- WordSim-353: skipped ({ws.get('reason')})")
            else:
                lines.append(f"- WordSim-353: n={ws.get('n')}, Spearman ρ={ws.get('spearman_rho'):.4f}")
            for gkey, gval in b.items():
                if not gkey.startswith("google_"):
                    continue
                if gval.get("skipped"):
                    lines.append(f"- Google ({gkey}): skipped ({gval.get('reason')})")
                else:
                    k = gval.get("k", "?")
                    top1 = gval.get("top1_acc")
                    topk = gval.get(f"top{k}_acc")
                    top1_s = f"{top1:.2f}%" if top1 is not None else "n/a"
                    topk_s = f"{topk:.2f}%" if topk is not None else "n/a"
                    lines.append(
                        f"- Google: n={gval.get('n')}, top-1={top1_s}, top-{k}={topk_s}"
                    )
            lines.append("")

    if payload.get("spot_checks"):
        lines.append("## Spot checks")
        lines.append("")
        for opt, sc in payload["spot_checks"].items():
            lines.append(f"### {DISPLAY_NAME.get(opt, opt)}")
            lines.append("")
            lines.append("**Similar words**")
            lines.append("")
            for item in sc.get("similar", []):
                if "error" in item:
                    lines.append(f"- `{item['word']}`: {item['error']}")
                else:
                    neigh = ", ".join(f"{n['word']} ({n['score']:.3f})" for n in item["neighbors"])
                    lines.append(f"- `{item['word']}` → {neigh}")
            lines.append("")
            lines.append("**Analogies** (`a:b :: c:?`)")
            lines.append("")
            for item in sc.get("analogies", []):
                q = item["query"]
                if "error" in item:
                    lines.append(f"- `{q[0]}:{q[1]} :: {q[2]}:?` — {item['error']}")
                else:
                    ans = ", ".join(f"{r['word']} ({r['score']:.3f})" for r in item["results"])
                    lines.append(f"- `{q[0]}:{q[1]} :: {q[2]}:?` → {ans}")
            lines.append("")

    lines.append("## Plots")
    lines.append("")
    titles = {
        "plots/train_loss_vs_epoch.png": "Training loss vs epoch",
        "plots/train_loss_vs_time.png": "Training loss vs time",
        "plots/test_loss_vs_epoch.png": "Test loss vs epoch",
        "plots/grad_w00.png": "Gradient of W[0][0]",
    }
    for p in plot_paths:
        lines.append(f"### {titles.get(p, p)}")
        lines.append("")
        lines.append(f"![{titles.get(p, p)}]({p})")
        lines.append("")

    if payload.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in payload["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (ncpu.floating, ncpu.integer)):
        return obj.item()
    if isinstance(obj, ncpu.ndarray):
        return obj.tolist()
    try:
        import cupy as cp

        if isinstance(obj, cp.ndarray):
            return cp.asnumpy(obj).tolist()
        if isinstance(obj, (cp.floating, cp.integer)):
            return float(obj) if isinstance(obj, cp.floating) else int(obj)
    except Exception:
        pass
    return str(obj)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(OUTPUT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")

    notes: List[str] = []
    hw = backend_info()
    print(f"Backend: {hw['array_module']}; GPU: {hw.get('gpu')}")

    config = {
        "BATCH_SIZE": BATCH_SIZE,
        "EMBED_DIM": EMBED_DIM,
        "EPOCHS": EPOCHS,
        "VOCAB_SIZE": VOCAB_SIZE,
        "CONTEXT_WINDOW_SIZE": CONTEXT_WINDOW_SIZE,
        "TEXT_CHARS": TEXT_CHARS,
        "SEED": SEED,
        "OPTIMIZERS": list(OPTIMIZERS),
        "LR": LR,
        "LR_DECAY": LR_DECAY,
        "RUN_BENCHMARKS": RUN_BENCHMARKS,
        "RUN_SPOT_CHECKS": RUN_SPOT_CHECKS,
        "SAMPLE_INTERVAL_SEC": SAMPLE_INTERVAL_SEC,
        "CORPUS_PATH": CORPUS_PATH,
        "SIMLEX_PATH": SIMLEX_PATH,
        "WORDSIM_PATH": WORDSIM_PATH,
        "GOOGLE_PATH": GOOGLE_PATH,
        "GOOGLE_TOP_K": list(GOOGLE_TOP_K),
    }

    t0 = time.time()
    print("Loading corpus...")
    txt = load_text(CORPUS_PATH, TEXT_CHARS)
    print(f"Corpus chars: {len(txt)}")
    print("Building dataset...")
    ds = build_dataset(txt, VOCAB_SIZE, CONTEXT_WINDOW_SIZE, SEED)
    print(f"vocab={ds['vocab_sz']} train={ds['n_train']} test={ds['n_test']}")

    sampler = ResourceSampler(SAMPLE_INTERVAL_SEC)
    sampler.set_label("setup")
    sampler.start()

    networks: Dict[str, list] = {}
    loss_history: Dict[str, list] = {}
    grad_history: Dict[str, list] = {}
    runtime: Dict[str, list] = {}
    test_losses: Dict[str, list] = {}
    results: Dict[str, Any] = {}
    benchmarks: Dict[str, Any] = {}
    spot_checks: Dict[str, Any] = {}

    try:
        for optim in OPTIMIZERS:
            print("=" * 60)
            print(f"Training optimizer: {optim}")
            sampler.set_label(optim)
            lr = LR.get(optim, LR.get("default", 0.01))
            decay = LR_DECAY.get(optim, LR_DECAY.get("default", 1.0))
            Nh, hist, ghist, times = build_net(
                optim,
                EMBED_DIM,
                EPOCHS,
                BATCH_SIZE,
                lr,
                decay,
                ds["vocab_sz"],
                CONTEXT_WINDOW_SIZE,
                ds["X_train"],
                ds["Y_train"],
            )
            networks[optim] = Nh
            loss_history[optim] = hist
            grad_history[optim] = ghist
            runtime[optim] = times

            # Per-epoch test loss
            sampler.set_label(f"{optim}_eval")
            tl = []
            for ep, N in enumerate(Nh):
                loss = eval_test_loss(
                    N, ds["X_test"], ds["Y_test"], ds["vocab_sz"], CONTEXT_WINDOW_SIZE, BATCH_SIZE
                )
                tl.append(loss)
                print(f"  [{optim}] epoch {ep + 1} test_loss={loss:.4f}")
            test_losses[optim] = tl
            results[optim] = {
                "train_loss": hist,
                "test_loss": tl,
                "runtime_sec": times,
                "lr": lr,
                "decay": decay,
            }

        if RUN_BENCHMARKS or RUN_SPOT_CHECKS:
            sampler.set_label("benchmarks")
            for optim in OPTIMIZERS:
                norm_embed = normalized_embedding(networks[optim][-1])
                if RUN_BENCHMARKS:
                    print(f"Benchmarks for {optim}...")
                    b: Dict[str, Any] = {
                        "simlex": run_simlex_benchmark(
                            SIMLEX_PATH, norm_embed, ds["vocab_st"], ds["word2idx"]
                        ),
                        "wordsim": run_wordsim_benchmark(
                            WORDSIM_PATH, norm_embed, ds["vocab_st"], ds["word2idx"]
                        ),
                    }
                    for k in GOOGLE_TOP_K:
                        b[f"google_top{k}"] = run_google_benchmark(
                            GOOGLE_PATH,
                            norm_embed,
                            ds["vocab"],
                            ds["vocab_st"],
                            ds["word2idx"],
                            k=k,
                        )
                    benchmarks[optim] = b
                    for key, val in b.items():
                        if val.get("skipped"):
                            notes.append(f"{optim}/{key}: {val.get('reason')}")
                if RUN_SPOT_CHECKS:
                    sc = {
                        "similar": [
                            get_similar(w, norm_embed, ds["vocab"], ds["word2idx"])
                            for w in SPOT_SIMILAR_WORDS
                        ],
                        "analogies": [
                            find_analogy(a, b, c, norm_embed, ds["vocab"], ds["word2idx"])
                            for a, b, c in SPOT_ANALOGIES
                        ],
                    }
                    spot_checks[optim] = sc
    except Exception:
        notes.append("Run aborted with exception; partial results may be incomplete.")
        notes.append(traceback.format_exc())
        print(traceback.format_exc())
        raise
    finally:
        sampler.stop()

    total_wall = time.time() - t0
    resources = {
        "overall": sampler.summarize(),
        "per_optimizer": {opt: sampler.summarize(opt) for opt in OPTIMIZERS},
        "raw_sample_count": len(sampler.samples),
    }

    plot_paths = save_plots(run_dir, loss_history, runtime, test_losses, grad_history, list(OPTIMIZERS))

    payload = {
        "run_id": run_id,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_wall_sec": total_wall,
        "config": config,
        "hardware": hw,
        "dataset": {
            "vocab_sz": ds["vocab_sz"],
            "n_tokens": ds["n_tokens"],
            "n_train": ds["n_train"],
            "n_test": ds["n_test"],
        },
        "results": results,
        "resources": resources,
        "benchmarks": benchmarks if RUN_BENCHMARKS else {},
        "spot_checks": spot_checks if RUN_SPOT_CHECKS else {},
        "plots": plot_paths,
        "notes": notes,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    write_report_md(run_dir, payload, plot_paths)
    print("=" * 60)
    print(f"Done in {total_wall:.1f}s")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"JSON:   {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
