"""Train the cross-attention scorer and score it against the same controls."""
import sys, json, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from peptide_eval import metrics
from peptide_eval.attention import CrossAttentionScorer
from peptide_eval.embed import device

DEV = device()
cohort = pd.read_parquet("data/processed/cohort_split.parquet")
tgt = np.load("data/processed/targets_residues.npz", allow_pickle=True)
pep = np.load("data/processed/peptides_residues.npz", allow_pickle=True)
TV, TO = tgt["vectors"], tgt["offsets"]; PV, PO = pep["vectors"], pep["offsets"]
tix = {s: i for i, s in enumerate(tgt["sequences"])}; pix = {s: i for i, s in enumerate(pep["sequences"])}

def target_block(seq):
    i = tix[seq]
    return torch.from_numpy(TV[TO[i]:TO[i+1]].astype(np.float32)).unsqueeze(0).to(DEV)

def peptide_block(seqs):
    mats = [PV[PO[pix[s]]:PO[pix[s]+1]] for s in seqs]
    L = max(m.shape[0] for m in mats)
    out = np.zeros((len(mats), L, 1280), dtype=np.float32)
    mask = np.zeros((len(mats), L), dtype=bool)
    for r, m in enumerate(mats):
        out[r, :m.shape[0]] = m; mask[r, :m.shape[0]] = True
    return torch.from_numpy(out).to(DEV), torch.from_numpy(mask).to(DEV)

def batches(frame, rng=None, cap=64):
    for seq, g in frame.groupby("target"):
        if rng is not None and len(g) > cap:
            g = g.iloc[rng.permutation(len(g))[:cap]]
        yield seq, g

@torch.no_grad()
def predict(model, frame, swap=None):
    model.eval(); out = {}
    for seq, g in batches(frame):
        t = target_block(swap[seq] if swap else seq)
        tm = torch.ones(1, t.shape[1], dtype=torch.bool, device=DEV)
        p, pm = peptide_block(list(g.peptide))
        out.update(zip(g.index, model(p, pm, t, tm).cpu().numpy()))
    return np.array([out[i] for i in frame.index])

def score(frame, pred):
    return metrics.per_target_spearman(frame.target.values, frame.affinity.values, pred)["macro_spearman"]

def train(use_target, seed, epochs=40, patience=8):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = CrossAttentionScorer(use_target=use_target).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    tr, va = cohort[cohort.split=="train"], cohort[cohort.split=="validation"]
    best, best_state, since = -2.0, None, 0
    for epoch in range(epochs):
        model.train()
        order = list(batches(tr, rng))
        for i in rng.permutation(len(order)):
            seq, g = order[i]
            t = target_block(seq); tm = torch.ones(1, t.shape[1], dtype=torch.bool, device=DEV)
            p, pm = peptide_block(list(g.peptide))
            y = torch.tensor(g.affinity.values, dtype=torch.float32, device=DEV)
            loss = torch.nn.functional.mse_loss(model(p, pm, t, tm), y)
            opt.zero_grad(); loss.backward(); opt.step()
        # Selected on validation ranking, not validation loss: the loss is
        # dominated by getting each target's overall scale right, which is the
        # part we already know no model does.
        v = score(va, predict(model, va)) or -2.0
        if v > best:
            best, best_state, since = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience: break
    model.load_state_dict(best_state)
    return model, best

test = cohort[cohort.split=="test"]
rng = np.random.default_rng(0)
tnames = sorted(test.target.unique())
swap = dict(zip(tnames, rng.permutation(tnames)))   # each target gets a different target's residues

rows = []
print(f"{'model':28s} {'val':>7s} {'per-target':>11s} {'enrich':>8s} {'pooled':>8s} {'sec':>6s}")
for label, use_target in (("cross-attention", True), ("peptide only (ablated)", False)):
    for seed in (0, 1, 2):
        t0 = time.time()
        model, val = train(use_target, seed)
        r = metrics.report(test.target.values, test.affinity.values, predict(model, test))
        rows.append({"model": label, "seed": seed, "val": val, **r})
        print(f"{label+' s'+str(seed):28s} {val:+7.3f} {r['macro_spearman']:+11.3f} "
              f"{r['macro_enrichment']:8.2f} {r['pooled_spearman']:+8.3f} {time.time()-t0:6.0f}")
        if use_target:
            sp = score(test, predict(model, test, swap=swap))
            print(f"{'  -> with WRONG target':28s} {'':7s} {sp:+11.3f}   <- should collapse if the target is used")
json.dump(rows, open("results/attention_test.json","w"), indent=2, default=float)

for label in ("cross-attention", "peptide only (ablated)"):
    v = np.array([r["macro_spearman"] for r in rows if r["model"] == label])
    print(f"\n{label:24s} mean {v.mean():+.3f}  sd {v.std():.3f}  seeds {np.round(v,3)}")
