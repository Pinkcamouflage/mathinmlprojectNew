"""
Interactive latent-space -> reward-tree recreator.

Companion to `landscape_surface.py`. That script lays a GRID x GRID landscape over
two RANDOM orthonormal latent directions (d1, d2), decoding every point
    z = center + a*d1 + b*d2
to a reward tree and scoring it. This tool lets you reach into that same landscape
and pull back the actual tree at any point you pick.

It prints, on startup, the exact two random directions used for a given --seed
(identical to landscape_surface, since both call `random_directions`), then drops
into a REPL. Enter a point and it decodes the tree, prints it, and reports the same
"proxy" fitness used by `--score proxy`.

Point input formats (one per line):
  a b              landscape coordinates: z = center + a*d1 + b*d2
                   (a runs along "random direction 1", b along "random direction 2"
                    — exactly the x/y axes of the surface_*.png plots)
  z: v1 v2 ... vL  a raw latent vector (L = model.latent numbers)
  dirs             reprint the two random directions
  help             reprint this help
  quit / q         exit

CLI knobs mirror landscape_surface so the geometry lines up:
  --seed    random-direction seed            (default 0)
  --center  origin | data | empty | best     (default empty: const(0) tree)
  --score   proxy | knn                      (default proxy)

The default center 'empty' anchors a=b=0 to where the null reward const(0)
encodes, so the landscape origin is a semantically empty reward.
"""
import os
import sys
import argparse

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                                       # noqa: E402
from visualize import load_model, encode_all                # noqa: E402
from landscape_surface import (                             # noqa: E402
    random_directions, good_bad_direction, make_proxy_scorer, make_knn_scorer,
    empty_tree_center, best_tree_center,
)
import data as data_mod                                     # noqa: E402


# ------------------------------------------------------------------
# Pretty-print a decoded SymbolicNode as an indented tree
# ------------------------------------------------------------------
def ascii_tree(node, prefix="", is_last=True):
    """Box-drawing rendering of a SymbolicNode (root at top)."""
    connector = "└── " if is_last else "├── "
    if node.value == "const":
        label = f"const={node.const_val:.3f}"
    else:
        label = node.value
    lines = [prefix + connector + label]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, c in enumerate(node.children):
        lines += ascii_tree(c, child_prefix, i == len(node.children) - 1)
    return lines


def print_tree(node):
    # Render with a fake root connector so the top line aligns nicely.
    lines = ascii_tree(node, prefix="", is_last=True)
    # drop the leading "└── " on the root line for a cleaner top
    lines[0] = lines[0].replace("└── ", "", 1)
    print("\n".join("    " + ln for ln in lines))


def fmt_vec(v, width=8):
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


def print_directions(d1, d2, seed, names, desc):
    print(f"\nLatent directions for --seed {seed} (latent dim = {len(d1)}):")
    print(desc)
    print(f"\n  d1 ({names[0]}):\n    {fmt_vec(d1)}")
    print(f"\n  d2 ({names[1]}):\n    {fmt_vec(d2)}")
    print(f"\n  check: |d1|={np.linalg.norm(d1):.4f}  |d2|={np.linalg.norm(d2):.4f}  "
          f"d1·d2={float(d1 @ d2):+.2e}\n")


RANDOM_DESC = (
    "  Two orthonormal vectors (same as landscape_surface.py): draw two standard-\n"
    "  normal vectors, normalize d1, Gram-Schmidt + normalize d2 (numpy rng(seed)).")
GOODBAD_DESC = (
    "  d1 = logistic boundary normal separating top-10%% from bottom-10%% trees,\n"
    "  pointing toward higher fitness (+a => better trees). d2 = orthogonal random.")


HELP = """
Enter a point in one of these forms:
  a b              landscape coords -> z = center + a*d1 + b*d2
  z: v1 v2 ... vL  a raw latent vector (L numbers)
  dirs             reprint the two random directions
  help             show this help
  quit / q         exit
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="random-direction seed (matches landscape_surface)")
    ap.add_argument("--direction", choices=["random", "good-bad"], default="random",
                    help="axes: two random orthonormal dirs, or the good->bad "
                         "logistic-separating axis (d1) + orthogonal random d2")
    ap.add_argument("--center", choices=["origin", "data", "empty", "best"], default="empty",
                    help="landscape center: origin | data mean | empty const(0) tree | "
                         "best logged tree")
    ap.add_argument("--score", choices=["proxy", "knn", "none"], default="proxy")
    ap.add_argument("--sample", action="store_true",
                    help="sample node types instead of argmax when decoding")
    ap.add_argument("--dump-dirs", metavar="PATH",
                    help="also save d1,d2 (and center) to a .npz file")
    args = ap.parse_args()

    model = load_model()
    latent = model.latent
    if args.direction == "good-bad":
        print("computing good->bad separating axis (encoding logged trees) ...")
        d1, d2 = good_bad_direction(latent, args.seed, model=model)
        dir_names, dir_desc = ("good->bad axis", "orthogonal random"), GOODBAD_DESC
    else:
        d1, d2 = random_directions(latent, args.seed)
        dir_names, dir_desc = ("random direction 1", "random direction 2"), RANDOM_DESC

    center = np.zeros(latent, dtype=np.float32)
    if args.center == "data":
        center = encode_all(model, data_mod.load_samples()).mean(0).astype(np.float32)
        print(f"center = data mean of encoded trees ({len(center)}-dim)")
    elif args.center == "empty":
        center = empty_tree_center(model)
        print(f"center = empty const(0) tree (latent norm {np.linalg.norm(center):.3f})")
    elif args.center == "best":
        center = best_tree_center(model)
        print(f"center = best logged tree (latent norm {np.linalg.norm(center):.3f})")
    else:
        print("center = origin")

    print_directions(d1, d2, args.seed, dir_names, dir_desc)

    if args.dump_dirs:
        np.savez(args.dump_dirs, d1=d1, d2=d2, center=center, seed=args.seed)
        print(f"saved directions -> {args.dump_dirs}")

    # Scorer (same notions as landscape_surface --score)
    scorer = score_label = None
    knn_grid = None
    if args.score == "proxy":
        scorer = make_proxy_scorer()
        score_label = "proxy fitness (corr. with true HalfCheetah objective)"
    elif args.score == "knn":
        knn_grid = make_knn_scorer(model)
        score_label = "kNN fitness (distance-weighted logged returns)"

    print(HELP)

    def decode_and_report(z, source):
        z_t = torch.from_numpy(z.astype(np.float32))
        tree = model.generate(z_t, sample=args.sample)
        print(f"\n[{source}]")
        print(f"  z = {fmt_vec(z)}")
        print(f"  repr: {repr(tree)}")
        print("  tree:")
        print_tree(tree)
        if args.score == "proxy":
            print(f"  {score_label}: {scorer(tree):+.4f}")
        elif args.score == "knn":
            val = float(knn_grid(z[None, :].astype(np.float64))[0])
            print(f"  {score_label}: {val:+.2f}")
        print()

    while True:
        try:
            line = input("point> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "q", "exit"):
            break
        if low == "help":
            print(HELP)
            continue
        if low == "dirs":
            print_directions(d1, d2, args.seed, dir_names, dir_desc)
            continue

        try:
            if low.startswith("z:"):
                nums = [float(x) for x in line[2:].replace(",", " ").split()]
                if len(nums) != latent:
                    print(f"  ! expected {latent} numbers for a raw latent vector, got {len(nums)}")
                    continue
                z = np.array(nums, dtype=np.float32)
                decode_and_report(z, "raw latent z")
            else:
                parts = [float(x) for x in line.replace(",", " ").split()]
                if len(parts) != 2:
                    print("  ! enter two numbers 'a b', or 'z: ...' for a raw vector "
                          "(type 'help')")
                    continue
                a, b = parts
                z = (center + a * d1 + b * d2).astype(np.float32)
                decode_and_report(z, f"landscape coords a={a:+.3f}, b={b:+.3f}")
        except ValueError as e:
            print(f"  ! could not parse numbers: {e}")


if __name__ == "__main__":
    main()
