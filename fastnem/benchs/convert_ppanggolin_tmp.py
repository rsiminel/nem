#!/usr/bin/env python3

import argparse
from pathlib import Path


def load_column_org_file(path):
    line = path.read_text().strip()
    return [tok.strip('"') for tok in line.split()]


def convert(tmp_dir, out_dir):
    tmp_dir = Path(tmp_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_path = tmp_dir / "nem_file.dat"
    nei_path = tmp_dir / "nem_file.nei"
    org_path = tmp_dir / "column_org_file"
    for p in (dat_path, nei_path, org_path):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    orgs = load_column_org_file(org_path)
    n_org = len(orgs)

    n_families = 0
    with open(dat_path) as dat_in, open(out_dir / "presence.csv", "w") as csv_out:
        for line in dat_in:
            row = line.strip().split("\t")
            if not row or row == [""]:
                continue
            if len(row) != n_org:
                raise SystemExit(f"row {n_families} has {len(row)} columns, "
                                  f"expected {n_org} organisms from column_org_file")
            csv_out.write(",".join(row) + "\n")
            n_families += 1

    n_edges = 0
    total_weight = 0.0
    with open(nei_path) as nei_in, open(out_dir / "graph.txt", "w") as graph_out:
        first = nei_in.readline()
        if first.strip() != "1":
            raise SystemExit(f"unexpected nem_file.nei header: {first!r}")
        for line in nei_in:
            tokens = line.strip().split("\t")
            if not tokens or tokens == [""]:
                continue
            fam_idx = int(tokens[0])
            k = int(tokens[1])
            if k == 0:
                continue
            neighbor_idx = tokens[2:2 + k]
            weights = tokens[2 + k:2 + 2 * k]
            for nb, w in zip(neighbor_idx, weights):
                graph_out.write(f"{fam_idx - 1} {int(nb) - 1} {w}\n")
                total_weight += float(w)
                n_edges += 1

    print(f"wrote {out_dir / 'presence.csv'} ({n_families} x {n_org}) and "
          f"{out_dir / 'graph.txt'} ({n_edges} directed edges)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmp-dir", required=True, type=str,
                     help="directory containing nem_file.dat/.nei and column_org_file")
    ap.add_argument("--out-dir", required=True, type=str,
                     help="where to write presence.csv and graph.txt")
    args = ap.parse_args()
    convert(args.tmp_dir, args.out_dir)


if __name__ == "__main__":
    main()
