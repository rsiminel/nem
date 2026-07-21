import numpy as np

from pyfastnem import _core

_DTYPE = {"f64": np.float64, "f32": np.float32}
_IMPL = {"f64": _core.partition_pangenome_f64, "f32": _core.partition_pangenome_f32}


def _graph_to_edges(graph, n_nodes, dtype):
    """
    Converts a networkx Graph/DiGraph to (from, to, weight) arrays.
    """
    if graph.number_of_nodes() != n_nodes:
        raise ValueError(f"graph has {graph.number_of_nodes()} nodes but presence has "
                          f"{n_nodes} gene families -- they must match one-to-one by index")

    directed = graph.is_directed()
    n_edges = graph.number_of_edges() * (1 if directed else 2)

    edges_from = np.empty(n_edges, dtype=np.int64)
    edges_to = np.empty(n_edges, dtype=np.int64)
    edges_weight = np.empty(n_edges, dtype=dtype)

    e = 0
    for i, j, w in graph.edges(data="weight", default=1.0):
        edges_from[e] = i
        edges_to[e] = j
        edges_weight[e] = w
        e += 1
        if not directed:
            edges_from[e] = j
            edges_to[e] = i
            edges_weight[e] = w
            e += 1

    return edges_from, edges_to, edges_weight


def partition_pangenome(presence, graph, K=3, beta=2.5, free_dispersion=False, sm_degree=10,
                         max_iter=100, tol=0.01, genome_weights=None, completeness=None,
                         precision="f64", threads=1):
    """
    Same semantics as pynem.partition_pangenome

    Two additions:

    precision : {"f64", "f32"}
        Floating-point type used internally (default "f64", matching pynem).
    threads : int
        Number of worker threads (default 1).

    Returns the same dict shape as pynem.partition_pangenome minus the "model" key
    """
    if precision not in _DTYPE:
        raise ValueError(f"precision must be 'f64' or 'f32', got {precision!r}")
    dtype = _DTYPE[precision]
    impl = _IMPL[precision]

    presence = np.ascontiguousarray(presence, dtype=dtype)
    n_genes, n_org = presence.shape

    edges_from, edges_to, edges_weight = _graph_to_edges(graph, n_genes, dtype)

    gw = None if genome_weights is None else np.ascontiguousarray(genome_weights, dtype=dtype)
    comp = None if completeness is None else np.ascontiguousarray(completeness, dtype=dtype)
    for name, arr in (("genome_weights", gw), ("completeness", comp)):
        if arr is not None and arr.shape != (n_org,):
            raise ValueError(f"{name} must have shape ({n_org},), got {arr.shape}")

    raw = impl(presence, edges_from, edges_to, edges_weight, n_genes, K, float(beta),
               bool(free_dispersion), sm_degree, max_iter, float(tol), threads, gw, comp)

    k, n_org_out = raw["k"], raw["n_org"]
    result = {
        "partition": np.array(list(raw["partition"])),
        "membership": np.asarray(raw["membership_flat"], dtype=dtype).reshape(n_genes, k),
        "labels_index": np.asarray(raw["labels_index"]),
        "centers": np.asarray(raw["centers_flat"], dtype=dtype).reshape(k, n_org_out),
        "dispersions": np.asarray(raw["dispersions_flat"], dtype=dtype).reshape(k, n_org_out),
        "proportions": np.asarray(raw["proportions"], dtype=dtype),
        "beta": raw["beta"],
        "n_iter": raw["n_iter"],
        "criteria": dict(zip("UDGLM", raw["criteria"])),
        "completeness": np.asarray(raw["completeness"], dtype=dtype) if "completeness" in raw else None,
    }
    return result
