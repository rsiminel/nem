# pyfastnem

```python
import networkx as nx
from pyfastnem import partition_pangenome

result = partition_pangenome(presence, graph, K=3, beta=2.5, precision="f64", threads=8)
```

Mirrors `pynem.partition_pangenome(X, G, ...)` signature and result.

- `precision`: `"f64"` (default, matches pynem) or `"f32"`
- `threads`: number of worker threads (default 1)

## Build

```bash
CXX=g++ pip install .
```
