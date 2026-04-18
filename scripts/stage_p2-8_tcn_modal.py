# ABOUTME: Stage P2-8 — TCN training on Modal GPU (T4) for 12 universe/target combos.
# ABOUTME: Inputs: output/sequences_{hk,kr,index_hk,index_kr}.npz  Outputs: output/predictions_tcn_*.csv, output/training_log_tcn_*.csv
# Run: source .venv/bin/activate && modal run scripts/stage_p2-8_tcn_modal.py

import modal
import os
import io
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# ---------------------------------------------------------------------------
# Modal app + volumes
# ---------------------------------------------------------------------------
app = modal.App("tcn-p2-8")

data_vol    = modal.Volume.from_name("tcn-p2-8-data",    create_if_missing=True)
results_vol = modal.Volume.from_name("tcn-p2-8-results", create_if_missing=True)

LOCAL_OUTPUT = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2/output")
LOCAL_LOGS   = Path("/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2/logs")

UNIVERSES = ["main_hk", "main_kr", "index_hk", "index_kr"]
TARGETS   = ["gap", "intraday", "cc"]

NPZ_MAP = {
    "main_hk":   "sequences_hk.npz",
    "main_kr":   "sequences_kr.npz",
    "index_hk":  "sequences_index_hk.npz",
    "index_kr":  "sequences_index_kr.npz",
}

STATIC_DIM = {
    "main_hk":  9,
    "main_kr":  9,
    "index_hk": 6,
    "index_kr": 6,
}

# Step-1 settings (start here — T_max ~1700-2200 is large)
N_BLOCKS     = 2       # drop dilation=16 block
FILTERS      = 64
KERNEL_SIZE  = 7
DROPOUT      = 0.2
BATCH_SIZE   = 128
MAX_EPOCHS   = 15
PATIENCE     = 5
LR           = 1e-3
LR_MIN       = 1e-5
WEIGHT_DECAY = 1e-4
MIN_TRAIN_DAYS = 252

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.13.0",
    )
)

# ---------------------------------------------------------------------------
# Training function (runs on Modal GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    volumes={
        "/data":    data_vol,
        "/results": results_vol,
    },
    retries=0,
)
def train_tcn(universe: str, target: str) -> dict:
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as _pd
    from scipy.stats import spearmanr
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{universe}/{target}] device={device}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    npz_path = f"/data/{NPZ_MAP[universe]}"
    d = np.load(npz_path, allow_pickle=True)
    sequences    = d["sequences"].astype(np.float32)       # (N, T_max, 15)
    masks        = d["masks"].astype(np.float32)           # (N, T_max)
    static_feats = d["static_features"].astype(np.float32) # (N, S)
    targets_arr  = d[f"targets_{target}"].astype(np.float32)
    dates        = d["dates"]
    tickers      = d["tickers"]

    N, T_max, F = sequences.shape
    S = static_feats.shape[1]
    print(f"  N={N} T_max={T_max} F={F} S={S}")

    # -----------------------------------------------------------------------
    # Model definition
    # -----------------------------------------------------------------------
    class CausalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  padding=pad, dilation=dilation)
            self.bn   = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self._pad = pad

        def forward(self, x):
            out = self.conv(x)
            if self._pad > 0:
                out = out[:, :, :-self._pad]
            out = self.drop(self.relu(self.bn(out)))
            return out + self.res(x)

    class TCN(nn.Module):
        def __init__(self, in_features, static_dim, n_blocks, filters, kernel_size, dropout):
            super().__init__()
            dilations = [1, 4, 16][:n_blocks]
            blocks = []
            in_ch = in_features
            for dil in dilations:
                blocks.append(CausalBlock(in_ch, filters, kernel_size, dil, dropout))
                in_ch = filters
            self.tcn = nn.Sequential(*blocks)
            fc_in = filters + static_dim
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, seq, mask, static):
            # seq: (B, T, F) -> (B, F, T)
            x = seq.permute(0, 2, 1)
            x = self.tcn(x)                       # (B, filters, T)
            m = mask.unsqueeze(1)                  # (B, 1, T)
            pooled = (x * m).sum(dim=2) / (m.sum(dim=2) + 1e-8)
            combined = torch.cat([pooled, static], dim=1)
            return self.fc(combined).squeeze(1)

    # -----------------------------------------------------------------------
    # Walk-forward splits (monthly rebalance, expanding window)
    # -----------------------------------------------------------------------
    unique_dates = sorted(np.unique(dates))
    n_dates = len(unique_dates)

    date_series = _pd.to_datetime(unique_dates)
    month_starts = []
    prev_m = None
    for i, dt in enumerate(date_series):
        m = (dt.year, dt.month)
        if m != prev_m:
            month_starts.append(i)
            prev_m = m

    folds = []
    for fi in range(len(month_starts)):
        test_start = month_starts[fi]
        if test_start < MIN_TRAIN_DAYS:
            continue
        test_end = month_starts[fi + 1] if (fi + 1) < len(month_starts) else n_dates
        train_date_set = set(unique_dates[:test_start])
        test_date_set  = set(unique_dates[test_start:test_end])
        folds.append((train_date_set, test_date_set, unique_dates[test_start], fi))

    print(f"  folds (monthly, min_train={MIN_TRAIN_DAYS}): {len(folds)}")

    # -----------------------------------------------------------------------
    # Resume from checkpoint if present
    # -----------------------------------------------------------------------
    import io as _io
    import json as _json

    partial_pred_fname = f"partial_tcn_{universe}_{target}.csv"
    partial_prog_fname = f"partial_progress_{universe}_{target}.json"

    all_preds   = []
    fold_logs   = []
    resume_from = 0

    try:
        prog_bytes = b"".join(results_vol.read_file(partial_prog_fname))
        prog = _json.loads(prog_bytes.decode())
        resume_from = int(prog.get("last_completed_fold_idx", -1)) + 1
        if resume_from > 0:
            pred_bytes_ck = b"".join(results_vol.read_file(partial_pred_fname))
            ck_df = _pd.read_csv(_io.BytesIO(pred_bytes_ck))
            all_preds = ck_df.to_dict(orient="records")
            fold_logs = prog.get("fold_logs", [])
            print(f"  [CHECKPOINT] resuming from fold {resume_from} "
                  f"({len(all_preds)} rows already accumulated)")
    except Exception:
        pass  # no checkpoint — start fresh

    # -----------------------------------------------------------------------
    # Walk-forward training
    # -----------------------------------------------------------------------
    for fold_i, (train_date_set, test_date_set, fold_date, fi) in enumerate(folds):
        if fold_i < resume_from:
            continue
        tr_rows = np.array([dates[i] in train_date_set for i in range(N)])
        te_rows = np.array([dates[i] in test_date_set  for i in range(N)])

        train_valid = tr_rows & ~np.isnan(targets_arr)
        test_valid  = te_rows & ~np.isnan(targets_arr)

        if train_valid.sum() < 64 or test_valid.sum() == 0:
            continue

        def make_tensors(idx_mask):
            idx = np.where(idx_mask)[0]
            return (
                torch.tensor(sequences[idx],    dtype=torch.float32),
                torch.tensor(masks[idx],        dtype=torch.float32),
                torch.tensor(static_feats[idx], dtype=torch.float32),
                torch.tensor(targets_arr[idx],  dtype=torch.float32),
                idx,
            )

        tr_seq, tr_mask, tr_stat, tr_y, tr_idx = make_tensors(train_valid)
        te_seq, te_mask, te_stat, te_y, te_idx = make_tensors(test_valid)

        # Standardize target
        mu_y  = tr_y.mean().item()
        std_y = tr_y.std().item() + 1e-8
        tr_y_norm = (tr_y - mu_y) / std_y

        # Standardize sequences on train masked positions
        tr_m_exp = tr_mask.unsqueeze(-1)                          # (N_tr, T, 1)
        seq_cnt  = tr_m_exp.sum(dim=(0, 1)).clamp(min=1)         # (15,)
        seq_mean = (tr_seq * tr_m_exp).sum(dim=(0, 1)) / seq_cnt # (15,)
        seq_sq   = ((tr_seq - seq_mean) ** 2 * tr_m_exp).sum(dim=(0, 1))
        seq_std  = (seq_sq / seq_cnt).sqrt().clamp(min=1e-8)

        tr_seq_n = (tr_seq - seq_mean) / seq_std
        te_seq_n = (te_seq - seq_mean) / seq_std

        # Standardize static
        stat_mean = tr_stat.mean(0)
        stat_std  = tr_stat.std(0).clamp(min=1e-8)
        tr_stat_n = (tr_stat - stat_mean) / stat_std
        te_stat_n = (te_stat - stat_mean) / stat_std

        # Split train into tr2 / val (90/10)
        n_total = len(tr_seq_n)
        n_val   = max(1, n_total // 10)
        n_tr2   = n_total - n_val

        ds_all = TensorDataset(tr_seq_n, tr_mask, tr_stat_n, tr_y_norm)
        ds_tr2, ds_val = torch.utils.data.random_split(
            ds_all, [n_tr2, n_val],
            generator=torch.Generator().manual_seed(42 + fold_i))
        dl_tr2 = DataLoader(ds_tr2, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model = TCN(F, S, N_BLOCKS, FILTERS, KERNEL_SIZE, DROPOUT).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_cnt  = 0
        best_state    = None
        epoch_done    = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for batch in dl_tr2:
                bseq, bmsk, bstat, by = [b.to(device) for b in batch]
                optimizer.zero_grad()
                loss_fn(model(bseq, bmsk, bstat), by).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for batch in dl_val:
                    bseq, bmsk, bstat, by = [b.to(device) for b in batch]
                    val_loss += loss_fn(model(bseq, bmsk, bstat), by).item()
                    n_val_batches += 1
            val_loss /= max(n_val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1

            epoch_done = epoch + 1
            if patience_cnt >= PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        # Batch predict
        def batch_predict(seq_t, mask_t, stat_t):
            preds = []
            with torch.no_grad():
                for i in range(0, len(seq_t), 256):
                    p = model(seq_t[i:i+256].to(device),
                              mask_t[i:i+256].to(device),
                              stat_t[i:i+256].to(device))
                    preds.append(p.cpu().numpy())
            return np.concatenate(preds)

        model.eval()
        te_preds_norm = batch_predict(te_seq_n, te_mask, te_stat_n)
        te_preds_raw  = te_preds_norm * std_y + mu_y

        tr_preds_norm = batch_predict(tr_seq_n, tr_mask, tr_stat_n)
        tr_preds_raw  = tr_preds_norm * std_y + mu_y

        # Metrics
        te_y_np = te_y.numpy()
        tr_y_np = tr_y.numpy()

        def safe_spearman(y, yhat):
            if len(y) < 2:
                return float("nan")
            r, _ = spearmanr(y, yhat)
            return float(r) if not np.isnan(r) else float("nan")

        def dir_acc(y, yhat):
            return float(np.mean(np.sign(y) == np.sign(yhat)))

        oos_ic   = safe_spearman(te_y_np, te_preds_raw)
        train_ic = safe_spearman(tr_y_np, tr_preds_raw)
        oos_dir  = dir_acc(te_y_np, te_preds_raw)

        # Collect predictions
        for j, idx in enumerate(te_idx):
            all_preds.append({
                "date":     str(dates[idx]),
                "ticker":   str(tickers[idx]),
                "y_pred":   float(te_preds_raw[j]),
                "y_actual": float(te_y_np[j]),
                "fold_id":  fold_i,
            })

        overfit_flag = ""
        if not (np.isnan(train_ic) or np.isnan(oos_ic)):
            if train_ic - oos_ic > 0.20:
                overfit_flag = "OVERFIT_FLAG"

        fold_logs.append({
            "fold_id":       fold_i,
            "fold_date":     fold_date,
            "n_train":       int(train_valid.sum()),
            "n_test":        int(test_valid.sum()),
            "train_ic":      train_ic,
            "oos_ic":        oos_ic,
            "oos_dir":       oos_dir,
            "best_val_loss": best_val_loss,
            "epochs_run":    epoch_done,
            "overfit_flag":  overfit_flag,
        })

        # --- per-fold checkpoint ---
        try:
            ck_pred_bytes = _pd.DataFrame(all_preds).to_csv(index=False).encode()
            ck_prog = _json.dumps({
                "last_completed_fold_idx": fold_i,
                "fold_logs": fold_logs,
            }).encode()
            with results_vol.batch_upload(force=True) as _batch:
                _batch.put_file(_io.BytesIO(ck_pred_bytes), partial_pred_fname)
                _batch.put_file(_io.BytesIO(ck_prog),       partial_prog_fname)
        except Exception as _ck_err:
            print(f"  [CHECKPOINT WARN] fold {fold_i}: {_ck_err}")

        print(f"  fold {fold_i:03d} {fold_date} n_tr={int(train_valid.sum()):5d} "
              f"n_te={int(test_valid.sum()):4d} train_ic={train_ic:+.3f} "
              f"oos_ic={oos_ic:+.3f} dir={oos_dir:.3f} ep={epoch_done} {overfit_flag}")

    # -----------------------------------------------------------------------
    # Save to results volume
    # -----------------------------------------------------------------------
    pred_df = _pd.DataFrame(all_preds)
    log_df  = _pd.DataFrame(fold_logs)

    pred_fname = f"predictions_tcn_{universe}_{target}.csv"
    log_fname  = f"training_log_tcn_{universe}_{target}.csv"

    pred_bytes = pred_df.to_csv(index=False).encode()
    log_bytes  = log_df.to_csv(index=False).encode()

    with results_vol.batch_upload(force=True) as batch:
        batch.put_file(_io.BytesIO(pred_bytes), pred_fname)
        batch.put_file(_io.BytesIO(log_bytes),  log_fname)

    # drop partial checkpoint markers now that final files are written
    try:
        results_vol.remove_file(partial_pred_fname)
    except Exception:
        pass
    try:
        results_vol.remove_file(partial_prog_fname)
    except Exception:
        pass

    mean_train_ic = float(log_df["train_ic"].mean()) if len(log_df) else float("nan")
    mean_oos_ic   = float(log_df["oos_ic"].mean())   if len(log_df) else float("nan")
    overfit_any   = bool((log_df["overfit_flag"] == "OVERFIT_FLAG").any()) if len(log_df) else False

    print(f"[{universe}/{target}] DONE rows={len(pred_df)} folds={len(log_df)} "
          f"mean_train_ic={mean_train_ic:+.4f} mean_oos_ic={mean_oos_ic:+.4f} "
          f"overfit={'OVERFIT_FLAG' if overfit_any else 'ok'}")

    return {
        "universe":      universe,
        "target":        target,
        "rows":          len(pred_df),
        "folds":         len(log_df),
        "mean_train_ic": mean_train_ic,
        "mean_oos_ic":   mean_oos_ic,
        "overfit_flag":  overfit_any,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    t_global = time.time()

    print("=== Stage P2-8: TCN Modal Training ===")
    print(f"Universes: {UNIVERSES}")
    print(f"Targets:   {TARGETS}")
    print(f"Settings:  Step-1 (N_BLOCKS={N_BLOCKS}, BATCH={BATCH_SIZE}, EPOCHS={MAX_EPOCHS})")

    # -----------------------------------------------------------------------
    # Step 1: Upload .npz to Modal volume (idempotent)
    # -----------------------------------------------------------------------
    print("\n[1/3] Uploading .npz files to Modal volume...")
    try:
        existing_entries = list(data_vol.listdir("/"))
        existing_names   = {e.path.lstrip("/") for e in existing_entries}
    except Exception:
        existing_names = set()

    files_to_upload = {}
    for universe, fname in NPZ_MAP.items():
        if fname in existing_names:
            print(f"  {fname} already in volume, skipping")
        else:
            local_path = LOCAL_OUTPUT / fname
            print(f"  uploading {fname} ({local_path.stat().st_size / 1e6:.1f} MB)...")
            files_to_upload[fname] = local_path

    if files_to_upload:
        with data_vol.batch_upload(force=True) as batch:
            for fname, local_path in files_to_upload.items():
                batch.put_file(str(local_path), fname)
        print(f"  uploaded {len(files_to_upload)} file(s)")
    else:
        print("  all files already in volume")

    # -----------------------------------------------------------------------
    # Step 2: Launch 6 MAIN configs only (index configs already done)
    # -----------------------------------------------------------------------
    main_configs = [
        ("main_hk", "gap"), ("main_hk", "intraday"), ("main_hk", "cc"),
        ("main_kr", "gap"), ("main_kr", "intraday"), ("main_kr", "cc"),
    ]
    all_configs = [(u, t) for u in UNIVERSES for t in TARGETS]  # for download phase

    print(f"\n[2/3] Launching {len(main_configs)} MAIN TCN runs in parallel on Modal T4...")
    t0 = time.time()
    raw_results = list(train_tcn.starmap(main_configs, return_exceptions=True))
    elapsed = time.time() - t0
    print(f"\n  Main runs finished in {elapsed/60:.1f} min ({elapsed/3600:.3f} h)")

    results = []
    for (u, t), res in zip(main_configs, raw_results):
        if isinstance(res, Exception) or (hasattr(res, "__class__") and "ExceptionWrapper" in type(res).__name__):
            print(f"  FAILED {u}/{t}: {res}")
        else:
            results.append(res)

    print(f"  Successful main runs: {len(results)}/{len(main_configs)}")

    # -----------------------------------------------------------------------
    # Step 3: Download results (all 12 configs including index)
    # -----------------------------------------------------------------------
    print("\n[3/3] Downloading results from Modal volume...")
    downloaded = []
    missing    = []

    for universe, target in all_configs:
        for prefix in ["predictions_tcn_", "training_log_tcn_"]:
            fname = f"{prefix}{universe}_{target}.csv"
            local = LOCAL_OUTPUT / fname
            try:
                data = b"".join(results_vol.read_file(fname))
                local.write_bytes(data)
                downloaded.append(fname)
            except Exception as e:
                print(f"  MISSING {fname}: {e}")
                missing.append(fname)

    print(f"  Downloaded {len(downloaded)} files, missing {len(missing)}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== SUMMARY ===")
    total_elapsed = time.time() - t_global
    approx_cost   = (elapsed / 3600) * 1.10 * len(main_configs)  # T4 parallel; Modal bills per container-second

    for universe, target in all_configs:
        fname = f"predictions_tcn_{universe}_{target}.csv"
        local = LOCAL_OUTPUT / fname
        result_row = next((r for r in results if r["universe"] == universe and r["target"] == target), {})
        if local.exists():
            df       = pd.read_csv(local)
            rows     = len(df)
            nan_pred = int(df["y_pred"].isna().sum())
            oos_ic   = result_row.get("mean_oos_ic", float("nan"))
            overfit  = "OVERFIT_FLAG" if result_row.get("overfit_flag") else ""
            print(f"  {universe}/{target}: rows={rows} oos_ic={oos_ic:+.4f} nan_pred={nan_pred} {overfit}")
        else:
            print(f"  {universe}/{target}: MISSING")

    print(f"\nFiles downloaded: {len(downloaded)}/24 (12 pred + 12 log)")
    pred_ok = sum(1 for u, t in all_configs if (LOCAL_OUTPUT / f"predictions_tcn_{u}_{t}.csv").exists())
    print(f"Prediction files: {pred_ok}/12")
    if missing:
        print(f"Missing: {missing}")
    print(f"Runtime adjustment: Step 1 (N_BLOCKS=2, BATCH_SIZE=128, MAX_EPOCHS=15)")
    print(f"Modal wall clock (parallel): {elapsed/60:.1f} min / {elapsed/3600:.3f} h")
    print(f"Approx Modal cost: ${approx_cost:.2f} (T4 parallel billing)")
    print(f"Total local elapsed: {total_elapsed/60:.1f} min")
    print("=== END P2-8 ===")
