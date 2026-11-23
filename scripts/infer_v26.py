"""CLI inference script for CAS-Canglong 0.25° & 0.25° Ensemble (16-month deterministic and 32-member ensemble).

Supports both the standalone `canglong.v2` module in this repository and `weatherai.models.canglong.load_official`.

Examples
--------
Deterministic 16-month 0.25° forecast from a normalized ERA5 tensor `(16, 8, 721, 1440)` or `(1, 8, 16, 721, 1440)`:

    python scripts/infer_v26.py \\
        --checkpoint canglong025_v26_16m_cslim.tar.gz \\
        --input normalized_window.pt \\
        --out forecast_det_16m.npz

32-member 0.25° Ensemble forecast (mean, spread, 5/25/75/95% quantiles):

    python scripts/infer_v26.py \\
        --checkpoint canglong025_v26_16m_ens.tar.gz \\
        --input normalized_window.pt \\
        --members 32 --leads 16 \\
        --out forecast_ens32_16m.npz

Quick self-test with synthetic input (no ERA5 needed):

    python scripts/infer_v26.py --checkpoint canglong025_v26_16m_ens.tar.gz --smoke-test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from canglong.v2 import load_canglong_v2 as _load_model
except ImportError:
    from weatherai.models.canglong import load_official as _load_model

SST_RAW_STD = 11.627
SST_RAW_MEAN = 286.79


def load_input_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path)
        t = torch.from_numpy(arr)
    else:
        t = torch.load(path, map_location="cpu", weights_only=True)
    if t.ndim == 4 and t.shape[0] == 16 and t.shape[1] == 8:
        t = t.permute(1, 0, 2, 3).unsqueeze(0)
    elif t.ndim == 4 and t.shape[0] == 8 and t.shape[1] == 16:
        t = t.unsqueeze(0)
    elif t.ndim == 5 and t.shape[1] == 8 and t.shape[2] == 16:
        pass
    else:
        raise ValueError(
            f"Expected shape (16, 8, 721, 1440), (8, 16, 721, 1440), or (B, 8, 16, 721, 1440), got {tuple(t.shape)}"
        )
    return t.float()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CAS-Canglong 0.25° / 0.25° Ensemble inference.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to .tar or .tar.gz checkpoint")
    ap.add_argument("--input", type=Path, default=None, help="Path to normalized ERA5 tensor (.pt or .npy)")
    ap.add_argument("--origin-month", type=int, default=None, help="Optional 1..12 calendar month of last input frame (auto-detected if omitted)")
    ap.add_argument("--leads", type=int, default=16, help="Forecast horizon in months (1..16, default 16)")
    ap.add_argument("--members", type=int, default=1, help="Ensemble members (1 = deterministic; >=2 = 0.25° Ensemble, e.g. 32)")
    ap.add_argument("--breed-cycles", type=int, default=1, help="Ensemble breeding cycles (default 1)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for ensemble perturbations")
    ap.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    ap.add_argument("--out", type=Path, default=None, help="Output .npz path")
    ap.add_argument("--smoke-test", action="store_true", help="Run a quick synthetic-input forward check")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    t0 = time.time()
    model = _load_model(str(args.checkpoint), map_location=device).eval()
    print(f"Loaded {args.checkpoint.name} on {device} in {time.time() - t0:.2f}s "
          f"(horizon={model.cfg.horizon}, cslim_v={model.cfg.cslim_version}, ensemble={model.cfg.has_ensemble})")

    if args.smoke_test:
        x = torch.zeros((1, 8, 16, 721, 1440), dtype=torch.float32, device=device)
        om = torch.tensor([8], dtype=torch.long, device=device)
    else:
        if args.input is None:
            ap.error("--input is required unless --smoke-test is set")
        x = load_input_tensor(args.input).to(device)
        om = None if args.origin_month is None else torch.full((x.shape[0],), args.origin_month - 1, dtype=torch.long, device=device)

    use_amp = device.type == "cuda"
    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        if args.members <= 1:
            y_norm = model(x, origin_month=om, out_leads=args.leads)  # (B, 1, L, 721, 1440)
            sst_k = (y_norm[:, 0].cpu().float().numpy() * SST_RAW_STD + SST_RAW_MEAN).astype(np.float32)
            print(f"Deterministic forward complete in {time.time() - t1:.2f}s | output shape: {sst_k.shape} | finite: {np.isfinite(sst_k).all()}")
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.out, sst_K=sst_k)
                print(f"Saved {args.out}")
        else:
            ens_norm = model.forecast_ensemble(
                x, n_members=args.members, origin_month=om, out_leads=args.leads,
                method="hybrid_obv", n_breed_cycles=args.breed_cycles, seed=args.seed,
                batch_size=1, output_device="cpu",
            )  # (B, M, 1, L, 721, 1440)
            ens_k = (ens_norm[:, :, 0].float().numpy() * SST_RAW_STD + SST_RAW_MEAN).astype(np.float32)
            mean_k = ens_k.mean(axis=1)
            std_k = ens_k.std(axis=1, ddof=1)
            p05_k = np.percentile(ens_k, 5.0, axis=1).astype(np.float32)
            p95_k = np.percentile(ens_k, 95.0, axis=1).astype(np.float32)
            print(f"{args.members}-member AI ensemble complete in {time.time() - t1:.2f}s | shape: {ens_k.shape} | mean spread L3: {std_k[0, min(2, args.leads-1)].mean():.3f} K")
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.out, sst_mean_K=mean_k, sst_std_K=std_k, sst_p05_K=p05_k, sst_p95_K=p95_k)
                print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
