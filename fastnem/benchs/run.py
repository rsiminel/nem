#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

def _load_data(data_dir, dtype=None):
    import networkx as nx
    import numpy as np

    X = np.loadtxt(Path(data_dir) / "presence.csv", delimiter=",", dtype=dtype or np.float64)
    edges = []
    max_node = -1
    with open(Path(data_dir) / "graph.txt") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            i, j = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) > 2 else 1.0
            edges.append((i, j, w))
            max_node = max(max_node, i, j)

    G = nx.DiGraph()
    G.add_nodes_from(range(max(max_node + 1, X.shape[0])))
    for (i, j, w) in edges:
        G.add_edge(i, j, weight=w)
    return X, G

def _pynem_worker(data_dir, pynem_root, k, beta, sm_degree, max_iter, tol, queue):
    try:
        if pynem_root:
            sys.path.insert(0, str(Path(pynem_root) / "src"))
        from pynem import partition_pangenome

        X, G = _load_data(data_dir)
        t0 = time.perf_counter()
        result = partition_pangenome(X, G, K=k, beta=beta, sm_degree=sm_degree,
                                      max_iter=max_iter, tol=tol)
        dt = time.perf_counter() - t0
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        queue.put({
            "time_s": dt, "n_iter": result["n_iter"], "mem_mib": peak_mib,
            "partition": "".join(result["partition"]),
            "dims": (X.shape[0], X.shape[1], G.number_of_edges()),
        })
    except Exception:
        import traceback
        queue.put({"error": traceback.format_exc()})


def _pyfastnem_worker(data_dir, k, beta, sm_degree, max_iter, tol, precision, threads, queue):
    try:
        import numpy as np
        from pyfastnem import partition_pangenome

        dtype = np.float32 if precision == "f32" else np.float64
        X, G = _load_data(data_dir, dtype=dtype)
        t0 = time.perf_counter()
        result = partition_pangenome(X, G, K=k, beta=beta, sm_degree=sm_degree, max_iter=max_iter,
                                      tol=tol, precision=precision, threads=threads)
        dt = time.perf_counter() - t0
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        queue.put({
            "time_s": dt, "n_iter": result["n_iter"], "mem_mib": peak_mib,
            "partition": "".join(result["partition"]),
        })
    except Exception:
        import traceback
        queue.put({"error": traceback.format_exc()})


def _run_isolated(target, args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    out = queue.get()
    proc.join()
    if "error" in out:
        raise RuntimeError(f"{target.__name__} failed:\n{out['error']}")
    if proc.exitcode != 0:
        raise RuntimeError(f"{target.__name__} exited with code {proc.exitcode}")
    return out


def agreement(a, b):
    if len(a) != len(b) or not a:
        return float("nan")
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return 100.0 * matches / len(a)


def print_block(n_families, d, edges, records):
    cols = [
        ("implementation", "impl", "<"),
        ("time (s)", "time_s", ">"),
        ("speedup", "speedup", ">"),
        ("mem (MiB)", "mem_mib", ">"),
        ("mem ratio", "mem_ratio", ">"),
        ("agreement", "agreement", ">"),
        ("n_iter", "n_iter", ">"),
    ]

    def fmt(rec, key):
        v = rec[key]
        if v is None:
            return "--"
        if key in ("speedup", "mem_ratio"):
            return f"{v:.1f}x"
        if key == "time_s":
            return f"{v:.3f}"
        if key == "mem_mib":
            return f"{v:.0f}"
        if key == "agreement":
            return f"{v:.1f}%"
        return str(v)

    widths = [max(len(title), max(len(fmt(r, key)) for r in records)) for title, key, _ in cols]

    print(f"\n== N={n_families}  D={d}  edges={edges} ==")
    header = "  ".join(title.ljust(w) if align == "<" else title.rjust(w)
                        for (title, _, align), w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in records:
        print("  ".join(fmt(r, key).ljust(w) if align == "<" else fmt(r, key).rjust(w)
                         for (_, key, align), w in zip(cols, widths)))


def run_dataset(data_dir, args):
    py = _run_isolated(_pynem_worker,
                        (str(data_dir), None, args.k, args.beta, args.sm_degree, args.max_iter,
                         args.tol))
    n, d, edges = py["dims"]

    old = None
    if args.pynem_old_root:
        old = _run_isolated(_pynem_worker,
                             (str(data_dir), args.pynem_old_root, args.k, args.beta,
                              args.sm_degree, args.max_iter, args.tol))

    if args.baseline == "pynem_old":
        if old is None:
            raise SystemExit("--baseline pynem_old requires --pynem-old-root")
        baseline = old
    else:
        baseline = py

    def rel(other):
        speedup = baseline["time_s"] / other["time_s"] if other["time_s"] > 0 else float("inf")
        mem_ratio = baseline["mem_mib"] / other["mem_mib"] if other["mem_mib"] > 0 else float("inf")
        agree_pct = agreement(baseline["partition"], other["partition"])
        return speedup, mem_ratio, agree_pct

    def row(impl, result, is_baseline):
        speedup, mem_ratio, agree_pct = (None, None, None) if is_baseline else rel(result)
        return {"impl": impl + " (baseline)" if is_baseline else impl, "time_s": result["time_s"],
                "n_iter": result["n_iter"], "mem_mib": result["mem_mib"], "speedup": speedup,
                "mem_ratio": mem_ratio, "agreement": agree_pct}

    records = []
    if old is not None:
        records.append(row("pynem_old", old, baseline is old))
    records.append(row("pynem", py, baseline is py))

    for prec in args.precisions:
        for t in args.threads:
            r = _run_isolated(_pyfastnem_worker,
                               (str(data_dir), args.k, args.beta, args.sm_degree, args.max_iter,
                                args.tol, prec, t))
            records.append(row(f"pyfastnem {prec} @{t}t", r, False))

    print_block(n, d, edges, records)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dirs", type=str, nargs="+", default=None,
                     help="one or more existing dirs already containing presence.csv + "
                          "graph.txt (e.g. from convert_ppanggolin_tmp.py")
    ap.add_argument("--pynem-old-root", type=str, default=None,
                     help="path to another pynem package root")
    ap.add_argument("--baseline", choices=["pynem", "pynem_old"], default="pynem",
                     help="which pynem baseline")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--sm-degree", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--precisions", type=str, nargs="+", default=["f64", "f32"],
                     choices=["f64", "f32"], help="precisions to bench")
    args = ap.parse_args()

    if args.data_dirs:
        for d in args.data_dirs:
            data_dir = Path(d)
            run_dataset(data_dir, args)

if __name__ == "__main__":
    main()
