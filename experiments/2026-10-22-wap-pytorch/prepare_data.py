r"""CROHME (WAP dağıtımı) BMP görüntülerini tek bir ``npz`` paketine çevirir.

Kaynak: https://github.com/JianshuZhang/WAP ``data/`` dizini (CROHME 2014 train 8835 / test 986,
CROHME 2016 test 1147 — yalnızca görüntü, etiket yok). Orijinal ``gen_pkl.py`` Python 2 ``cPickle``
kullanır; burada aynı sözlük yapısı (``{uid: uint8[1,H,W]}``) ``npz`` olarak yazılır.

Kullanım::

    python prepare_data.py --wap C:\aicls-data\third_party\WAP --out C:\aicls-data\datasets\crohme_wap
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_captions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue
        uid, label = line.split("\t", 1)
        out[uid] = label.strip()
    return out


def pack(image_dir: Path, uids: list[str] | None) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    if uids is None:
        uids = sorted(p.name[: -len("_0.bmp")] for p in image_dir.glob("*_0.bmp"))
    for uid in uids:
        im = Image.open(image_dir / f"{uid}_0.bmp").convert("L")
        feats[uid] = np.asarray(im, dtype=np.uint8)[None]
    return feats


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = args.wap / "data"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"source": "JianshuZhang/WAP data/", "splits": {}}
    for split, cap, img in (
        ("train", "train_caption.txt", "off_image_train"),
        ("test2014", "test_caption.txt", "off_image_test"),
    ):
        caps = read_captions(data / cap)
        feats = pack(data / img, list(caps))
        out = args.out / f"{split}.npz"
        np.savez_compressed(out, **feats)
        (args.out / f"{split}_caption.txt").write_text(
            "".join(f"{k}\t{v}\n" for k, v in caps.items()), encoding="utf-8"
        )
        manifest["splits"][split] = {
            "n": len(feats),
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
        }
        print(split, len(feats))
    img16 = data / "crohme16" / "off_image_test"
    if img16.is_dir():
        feats = pack(img16, None)
        out = args.out / "test2016_images.npz"
        np.savez_compressed(out, **feats)
        manifest["splits"]["test2016_images"] = {
            "n": len(feats),
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
            "note": "etiket yok; WAP deposunda yalnızca görüntü dağıtılıyor",
        }
        print("test2016_images", len(feats))
    (args.out / "dictionary.txt").write_bytes((data / "dictionary.txt").read_bytes())
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", args.out / "manifest.json")


if __name__ == "__main__":
    main()
