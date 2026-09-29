"""Fine-tune ESM-2's top layers under the cross-attention head, with the same controls."""
import sys, json, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from peptide_eval import metrics
from peptide_eval.finetune import FineTuneScorer
from peptide_eval.embed import device

DEV = device()
cohort = pd.read_parquet("data/processed/cohort_split.parquet")
train_f = cohort[cohort.split == "train"]
valid_f = cohort[cohort.split == "validation"]
test_f = cohort[cohort.split == "test"]
names = sorted(test_f.target.unique())
SWAP = dict(zip(names, np.random.default_rng(0).permutation(names)))

def rank(frame, pred):
    return metrics.per_target_spearman(frame.target.values, frame.affinity.values, pred)["macro_spearman"]

@torch.no_grad()
def predict(model, frame, swap=None):
    model.eval(); out = {}
    for seq, g in frame.groupby("target"):
        for start in range(0, len(g), 48):
            chunk = g.iloc[start : start + 48]
            scores = model(list(chunk.peptide), swap[seq] if swap else seq, DEV)
            out.update(zip(chunk.index, scores.float().cpu().numpy()))
    return np.array([out[i] for i in frame.index])

def run(use_target, seed, epochs=12, patience=3, cap=48, unfrozen=2, body_lr=1e-5):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = FineTuneScorer(unfrozen=unfrozen, use_target=use_target).to(DEV)
    # The head is randomly initialised and the encoder is not: one learning rate
    # either leaves the head untrained or destroys the pretrained layers.
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("esm.")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("esm.")]
    opt = torch.optim.AdamW([{"params": body, "lr": body_lr}, {"params": head, "lr": 3e-4}],
                            weight_decay=1e-2)
    groups = list(train_f.groupby("target"))
    best, state, since, history = -2.0, None, 0, []
    for epoch in range(epochs):
        model.train(); t0 = time.time()
        for i in rng.permutation(len(groups)):
            seq, g = groups[i]
            if len(g) > cap:
                g = g.iloc[rng.permutation(len(g))[:cap]]
            y = torch.tensor(g.affinity.values, dtype=torch.float32, device=DEV)
            loss = torch.nn.functional.mse_loss(model(list(g.peptide), seq, DEV), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        # Windowed long targets leave a lot of transient allocation behind, and
        # the validation pass needs room for its own.
        if DEV == "mps":
            torch.mps.empty_cache()
        v = rank(valid_f, predict(model, valid_f)) or -2.0
        if DEV == "mps":
            torch.mps.empty_cache()
        history.append(v)
        print(f"    epoch {epoch:2d}  val {v:+.3f}  {time.time()-t0:.0f}s", flush=True)
        if v > best:
            # Only the trainable tensors, and on the CPU. Snapshotting all 652M
            # parameters on the accelerator costs 2.6 GB per copy and holds two
            # at once while the new one is built, which pushes an 18 GB machine
            # into swap; 612M of those never change anyway.
            best, since = v, 0
            state = {k: t.detach().to("cpu").clone()
                     for k, t in model.named_parameters() if t.requires_grad}
        else:
            since += 1
            if since >= patience: break
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name in state:
                tensor.copy_(state[name].to(DEV))
    return model, best, history

# The first configuration overfits within two epochs, so the same question is
# asked again with a quarter of the trainable encoder and a fifth of the rate.
# Reporting only the aggressive setting would let a capacity failure stand in
# for an answer about whether the signal is reachable at all.
SETTINGS = [("2 layers, lr 1e-5", 2, 1e-5, 0), ("1 layer, lr 2e-6", 1, 2e-6, 0),
            ("1 layer, lr 2e-6", 1, 2e-6, 1)]
rows = []
for label, use_target in (("finetuned cross-attention", True), ("finetuned peptide-only", False)):
    for setting, unfrozen, body_lr, seed in SETTINGS:
        if not use_target and setting != "1 layer, lr 2e-6":
            continue
        print(f"\n{label}  [{setting}]  seed {seed}", flush=True)
        model, val, history = run(use_target, seed, unfrozen=unfrozen, body_lr=body_lr)
        report = metrics.report(test_f.target.values, test_f.affinity.values, predict(model, test_f))
        wrong = rank(test_f, predict(model, test_f, swap=SWAP)) if use_target else None
        rows.append({"model": label, "setting": setting, "seed": seed, "val": val,
                     "wrong_target": wrong, "epochs": len(history), **report})
        print(f"  -> [{setting} s{seed}] val {val:+.3f} | test per-target {report['macro_spearman']:+.3f} | "
              f"enrich {report['macro_enrichment']:.2f} | pooled {report['pooled_spearman']:+.3f}"
              + (f" | WRONG target {wrong:+.3f}" if wrong is not None else ""), flush=True)
        json.dump(rows, open("results/finetune_test.json", "w"), indent=2, default=float)
print("\ndone")
