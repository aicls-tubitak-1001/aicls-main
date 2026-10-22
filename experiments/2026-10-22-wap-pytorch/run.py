"""WAP-PyTorch: CROHME 2014 üzerinde eğitim + değerlendirme (docs/deneme-alani.md).

Orijinal ``train_nmt.py`` hiperparametreleri (``config.yaml``): Adam 2e-4, L2 1e-4, gradyan
kırpma 100, yığın <= 8 örnek **ve** <= 500 000 piksel (görüntüler alana göre sıralanıp kovalanır),
``maxlen`` 200, her dönem sonunda test kümesinde doğrulama, WER 15 dönem iyileşmezse lr/2.

Büyük çıktılar ``AICLS_DATA_ROOT/runs/wap-pytorch/``, özet ``results/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wap import WAP

from aicls.eval.metrics import edit_distance

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AICLS_DATA_ROOT", r"C:\aicls-data"))


def load_dict(path: Path) -> dict[str, int]:
    d: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        w, i = line.split()
        d[w] = int(i)
    return d


def load_split(root: Path, split: str, vocab: dict[str, int], maxlen: int, max_px: int):
    feats = np.load(root / f"{split}.npz")
    items = []
    for line in (root / f"{split}_caption.txt").read_text(encoding="utf-8").splitlines():
        uid, label = line.split("\t", 1)
        ids = [vocab[w] for w in label.split()]
        im = feats[uid]
        if len(ids) > maxlen or im.shape[1] * im.shape[2] > max_px:
            continue
        items.append((uid, im, ids, label))
    return items


def make_batches(items, batch_size: int, batch_px: int):
    """Orijinal ``dataIterator``: alana göre sırala, en büyük görüntü x n <= batch_px."""
    items = sorted(items, key=lambda t: t[1].shape[1] * t[1].shape[2])
    batches, cur, biggest = [], [], 0
    for it in items:
        size = it[1].shape[1] * it[1].shape[2]
        biggest = max(biggest, size)
        if cur and (biggest * (len(cur) + 1) > batch_px or len(cur) == batch_size):
            batches.append(cur)
            cur, biggest = [], size
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


def collate(batch, device):
    h = max(b[1].shape[1] for b in batch)
    w = max(b[1].shape[2] for b in batch)
    t = max(len(b[2]) for b in batch) + 1
    x = np.zeros((len(batch), 1, h, w), np.float32)
    xm = np.zeros((len(batch), h, w), np.float32)
    y = np.zeros((len(batch), t), np.int64)
    ym = np.zeros((len(batch), t), np.float32)
    for i, (_, im, ids, _) in enumerate(batch):
        x[i, :, : im.shape[1], : im.shape[2]] = im / 255.0
        xm[i, : im.shape[1], : im.shape[2]] = 1.0
        y[i, : len(ids)] = ids
        ym[i, : len(ids) + 1] = 1.0
    return tuple(torch.from_numpy(a).to(device) for a in (x, xm, y, ym))


@torch.no_grad()
def evaluate(model, items, device, inv, *, beam: int, batch_px: int, maxlen: int):
    model.eval()
    hyps: dict[str, list[str]] = {}
    if beam <= 1:
        for batch in make_batches(items, 8, batch_px):
            x, xm, _, _ = collate(batch, device)
            for (uid, *_), seq in zip(batch, model.greedy(x, xm, maxlen), strict=True):
                hyps[uid] = [inv[i] for i in seq]
    else:
        for uid, im, _, _ in items:
            x, xm, _, _ = collate([(uid, im, [], "")], device)
            hyps[uid] = [inv[i] for i in model.beam(x, xm, k=beam, maxlen=maxlen)]
    errs = n_tok = correct = 0
    for uid, _, _, label in items:
        ref = label.split()
        errs += edit_distance(ref, hyps[uid])
        n_tok += len(ref)
        correct += int(ref == hyps[uid])
    return {"wer": errs / n_tok, "exprate": correct / len(items), "n": len(items)}, hyps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None, help="duman testi için örnek sınırı")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tr = cfg["train"]
    epochs = args.epochs if args.epochs is not None else tr["epochs"]
    beam = args.beam if args.beam is not None else cfg["eval"]["beam"]
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = DATA_ROOT / cfg["data"]["root"]
    run_dir = DATA_ROOT / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_dict(data_root / "dictionary.txt")
    inv = {i: w for w, i in vocab.items()}
    train = load_split(data_root, "train", vocab, tr["maxlen"], tr["max_image_px"])
    test = load_split(data_root, "test2014", vocab, tr["maxlen"], tr["max_image_px"])
    if args.max_train:
        train = train[: args.max_train]
        test = test[: max(args.max_train // 8, 8)]
    print(f"train {len(train)}  test {len(test)}  device {device}", flush=True)

    m = cfg["model"]
    model = WAP(
        vocab=len(vocab),
        dim_word=m["dim_word"],
        dim_dec=m["dim_dec"],
        dim_att=m["dim_attention"],
        dim_cov=m["dim_coverage"],
        k_cov=m["kernel_coverage"],
        dropout=m["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.2f}M", flush=True)
    ckpt = run_dir / "best.pt"
    log_path = run_dir / "train_log.jsonl"
    history: list[dict] = []

    if not args.eval_only:
        opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
        lr = tr["lr"]
        best_wer, bad = math.inf, 0
        step = 0
        for ep in range(epochs):
            model.train()
            batches = make_batches(train, tr["batch_size"], tr["batch_image_px"])
            random.shuffle(batches)
            t0, tot, nb = time.time(), 0.0, 0
            for batch in batches:
                x, xm, y, ym = collate(batch, device)
                loss = model(x, xm, y, ym)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["clip"])
                opt.step()
                tot += loss.item()
                nb += 1
                step += 1
                if step % 100 == 0:
                    print(f"ep {ep} step {step} loss {tot / nb:.3f} lr {lr:g}", flush=True)
            metrics, _ = evaluate(
                model, test, device, inv, beam=1, batch_px=tr["batch_image_px"], maxlen=tr["maxlen"]
            )
            rec = {
                "epoch": ep,
                "step": step,
                "train_loss": tot / max(nb, 1),
                "lr": lr,
                "epoch_s": time.time() - t0,
                **{f"valid_{k}": v for k, v in metrics.items()},
            }
            history.append(rec)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            if metrics["wer"] < best_wer:
                best_wer, bad = metrics["wer"], 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad > tr["patience"]:
                    lr /= 2
                    for g in opt.param_groups:
                        g["lr"] = lr
                    bad = 0
                    print(f"lr decay -> {lr:g}", flush=True)
                    if lr < tr["lr"] / 16:
                        break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    t0 = time.time()
    metrics, hyps = evaluate(
        model, test, device, inv, beam=beam, batch_px=tr["batch_image_px"], maxlen=tr["maxlen"]
    )
    metrics["decode_s"] = time.time() - t0
    metrics["beam"] = beam
    (run_dir / f"test2014_decode_beam{beam}.txt").write_text(
        "".join(f"{uid}\t{' '.join(h)}\n" for uid, h in hyps.items()), encoding="utf-8"
    )
    summary = {
        "name": cfg["name"],
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "params": n_params,
        "n_train": len(train),
        "n_test": len(test),
        "epochs_run": len(history),
        "test2014": metrics,
        "history": history,
        "paper_reference": {"crohme2014_exprate": 0.4655, "crohme2016_exprate": 0.4455},
    }
    out = HERE / "results" / "summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
