"""
Train the tree-VAE.

1. Pretrain on a large synthetic corpus of random grammar-valid trees
   (symbolic_tree.generate_random_tree) so the model learns the tree manifold.
2. Fine-tune on the (deduplicated) real trees from bestRun/training.csv.

Checkpoint -> tree_vae/checkpoints/tree_vae.pt
"""
import os
import sys
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tv_config as C                          # noqa: E402
from symbolic_tree import generate_random_tree  # noqa: E402
from graph import tree_to_graph                 # noqa: E402
from model import TreeVAE                        # noqa: E402
import data as data_mod                          # noqa: E402


def make_graphs(trees):
    return [tree_to_graph(t, device=C.DEVICE) for t in trees]


def beta_for_epoch(epoch):
    if epoch >= C.KL_WARMUP_EPOCHS:
        return C.BETA_MAX
    return C.BETA_MAX * (epoch + 1) / C.KL_WARMUP_EPOCHS


def run_epoch(model, opt, graphs, beta, train=True):
    if train:
        model.train()
        random.shuffle(graphs)
    else:
        model.eval()

    tot_ce = tot_const = tot_kl = 0.0
    n = 0
    for start in range(0, len(graphs), C.BATCH_SIZE):
        batch = graphs[start:start + C.BATCH_SIZE]
        if train:
            opt.zero_grad()
        ce_b = const_b = kl_b = 0.0
        for g in batch:
            ce, const, kl, _, _ = model(g)
            loss = ce + const + beta * kl
            if train:
                loss.backward()
            ce_b += ce.item(); const_b += const.item(); kl_b += kl.item()
        if train:
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        tot_ce += ce_b; tot_const += const_b; tot_kl += kl_b; n += len(batch)
    return tot_ce / n, tot_const / n, tot_kl / n


def structural_repr(node):
    """repr with const values blanked out, so structural match ignores const precision."""
    if node.is_leaf():
        return "const" if node.value == "const" else node.value
    inner = ", ".join(structural_repr(c) for c in node.children)
    return f"{node.value}({inner})"


@torch.no_grad()
def recon_metrics(model, samples, n=300):
    """Honest reconstruction metrics from latent means.

      struct_acc : free-decode -> structural exact match (const values ignored)
      node_acc   : teacher-forced per-node type accuracy
      const_mae  : teacher-forced mean abs error on const leaves
    """
    model.eval()
    subset = samples[:n]
    struct_hits = 0
    nc = nt = 0          # node correct / total
    cerr = 0.0; ccount = 0
    for s in subset:
        g = tree_to_graph(s.tree, device=C.DEVICE)
        mu, _ = model.encode(g)
        out = model.generate(mu, sample=False)
        if structural_repr(out) == structural_repr(s.tree):
            struct_hits += 1
        c, t, e, k = model.teacher_forced_stats(mu, g)
        nc += c; nt += t; cerr += e; ccount += k
    return {
        "struct_acc": struct_hits / max(1, len(subset)),
        "node_acc":   nc / max(1, nt),
        "const_mae":  cerr / max(1, ccount),
    }


def train_phase(model, opt, graphs, epochs, label, samples_for_acc=None):
    for epoch in range(epochs):
        beta = beta_for_epoch(epoch)
        ce, const, kl = run_epoch(model, opt, graphs, beta, train=True)
        msg = f"[{label}] epoch {epoch+1}/{epochs}  ce={ce:.3f} const={const:.3f} kl={kl:.2f} beta={beta:.3f}"
        if samples_for_acc is not None and (epoch + 1) % 5 == 0:
            m = recon_metrics(model, samples_for_acc)
            msg += f"  struct_acc={m['struct_acc']:.3f} node_acc={m['node_acc']:.3f} const_mae={m['const_mae']:.3f}"
        print(msg)


def main():
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    torch.manual_seed(0); random.seed(0)

    print("Loading real trees from bestRun/training.csv ...")
    samples = data_mod.load_samples()
    uniq = data_mod.dedup(samples)
    print(f"  {len(samples)} trees, {len(uniq)} unique")

    print(f"Generating {C.SYNTH_CORPUS_SIZE} synthetic trees ...")
    synth = [generate_random_tree(C.MAX_TREE_DEPTH) for _ in range(C.SYNTH_CORPUS_SIZE)]

    model = TreeVAE().to(C.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)

    print("\n== Pretraining on synthetic corpus ==")
    train_phase(model, opt, make_graphs(synth), C.EPOCHS_PRETRAIN, "pretrain")

    print("\n== Fine-tuning on real trees ==")
    real_graphs = make_graphs([s.tree for s in uniq])
    train_phase(model, opt, real_graphs, C.EPOCHS_FINETUNE, "finetune", samples_for_acc=uniq)

    ckpt = os.path.join(C.CKPT_DIR, "tree_vae.pt")
    torch.save({"model": model.state_dict(),
                "hidden": C.HIDDEN_DIM, "latent": C.LATENT_DIM}, ckpt)
    print(f"\nsaved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
