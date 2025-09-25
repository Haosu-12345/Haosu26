import math
import os
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

set_seed(42)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 65
    lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.5
    d_model: int = 128
    n_heads: int = 4
    enc_layers: int = 2
    lstm_hidden: int = 64
    lstm_layers: int = 2
    seq_len: int = 50
    pred_horizon: int = 1
    quantiles: List[float] = (0.025, 0.1, 0.5, 0.9, 0.975)
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_model.pt"
    early_stop_patience: int = 25
    scaler: str = "minmax"  # or "standard"

CFG = TrainConfig()


class SeriesScaler:
    def __init__(self, mode="minmax"):
        self.mode = mode
        self.params = {}

    def fit(self, series: np.ndarray):
        if self.mode == "minmax":
            self.params["min"] = float(np.min(series))
            self.params["max"] = float(np.max(series))
        else:  # standard
            self.params["mean"] = float(np.mean(series))
            self.params["std"] = float(np.std(series) + 1e-12)

    def transform(self, series: np.ndarray) -> np.ndarray:
        if self.mode == "minmax":
            denom = (self.params["max"] - self.params["min"]) or 1.0
            return (series - self.params["min"]) / denom
        else:
            return (series - self.params["mean"]) / (self.params["std"] or 1.0)

    def inverse_transform(self, series: np.ndarray) -> np.ndarray:
        if self.mode == "minmax":
            return series * (self.params["max"] - self.params["min"]) + self.params["min"]
        else:
            return series * (self.params["std"]) + self.params["mean"]

def load_series_from_csv(path: str, target_col: str, date_col: str = None) -> pd.Series:
    
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

    s = df[target_col].astype(float).reset_index(drop=True)
    return s

def time_split_index(n: int, test_ratio: float) -> int:
    return int(np.floor(n * (1.0 - test_ratio)))

class SlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, horizon: int = 1):
        self.series = series.astype(np.float32)
        self.seq_len = seq_len
        self.h = horizon
        self.n = len(series)

    def __len__(self):
        return max(0, self.n - self.seq_len - self.h + 1)

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len + self.h - 1]
        x = torch.from_numpy(x).view(self.seq_len, 1)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class InformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.pe = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pe(x)
        x = self.encoder(x)
        return x


class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x + residual)
        return x


class GRQLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=2, dropout=0.5, quantiles=(0.025,0.1,0.5,0.9,0.975)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.grn = GRN(hidden_dim, hidden_dim * 2)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in quantiles])
        self.quantiles = quantiles

    def forward(self, enc_out):
        out, _ = self.lstm(enc_out)
        last = out[:, -1, :]
        last = self.grn(last)
        preds = [head(last) for head in self.heads]
        return preds


class GRQLSTMInformer(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.encoder = InformerEncoder(cfg.d_model, cfg.n_heads, cfg.enc_layers, cfg.dropout)
        self.decoder = GRQLSTM(cfg.d_model, cfg.lstm_hidden, cfg.lstm_layers, cfg.dropout, cfg.quantiles)
        self.quantiles = cfg.quantiles

    def forward(self, x):
        enc = self.encoder(x)
        preds = self.decoder(enc)
        return preds


def pinball_loss(preds_list: List[torch.Tensor], y: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    loss = 0.0
    for pred, q in zip(preds_list, quantiles):
        e = y - pred
        loss_q = torch.maximum((q - 1) * e, q * e).mean()
        loss = loss + loss_q
    return loss / len(quantiles)

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def metrics_point(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse + 1e-12)
    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-12)
    smape = np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "SMAPE": float(smape)}

def q_risk(y_true: np.ndarray, y_quant: np.ndarray, q: float) -> float:
    e = y_true - y_quant
    return float(np.mean(np.maximum((q - 1) * e, q * e)))

def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.05) -> float:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    penalty = (2.0 / alpha) * (lower - y_true) * below + (2.0 / alpha) * (y_true - upper) * above
    ws = width + penalty
    return float(np.mean(ws))


def train_one_epoch(model, loader, optimizer, cfg):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        optimizer.zero_grad()
        preds = model(x)
        loss = pinball_loss(preds, y, model.quantiles)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total = 0.0
    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        preds = model(x)
        loss = pinball_loss(preds, y, model.quantiles)
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def test_model(model, loader, cfg, scaler: SeriesScaler):
    model.eval()
    all_true = []
    all_q_preds: Dict[float, List[float]] = {q: [] for q in model.quantiles}

    for x, y in loader:
        x = x.to(cfg.device)
        preds = model(x)
        for q, p in zip(model.quantiles, preds):
            all_q_preds[q].extend(to_numpy(p).ravel().tolist())
        all_true.extend(to_numpy(y).ravel().tolist())

    y_true_scaled = np.array(all_true, dtype=np.float64)
    y_true = scaler.inverse_transform(y_true_scaled)

    q_preds_scaled = {q: np.array(vals, dtype=np.float64) for q, vals in all_q_preds.items()}
    q_preds = {q: scaler.inverse_transform(arr) for q, arr in q_preds_scaled.items()}

    y_pred_point = q_preds[0.5]
    det = metrics_point(y_true, y_pred_point)
    qrisk = {f"Q-Risk@{q:.3f}": q_risk(y_true, q_preds[q], q) for q in q_preds.keys()}
    ws = winkler_score(y_true, q_preds[0.025], q_preds[0.975], alpha=0.05)

    return det, qrisk, ws


def build_loaders(series: pd.Series, cfg: TrainConfig):
    values = series.values.astype(np.float64)
    n = len(values)
    test_start = time_split_index(n, cfg.test_ratio)
    trainval = values[:test_start]
    test = values[test_start:]
    val_start = time_split_index(len(trainval), cfg.val_ratio)
    train = trainval[:val_start]
    val = trainval[val_start:]

    scaler = SeriesScaler(CFG.scaler)
    scaler.fit(train)
    train_s = scaler.transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)

    ds_train = SlidingWindowDataset(train_s, cfg.seq_len, cfg.pred_horizon)
    ds_val = SlidingWindowDataset(val_s, cfg.seq_len, cfg.pred_horizon)
    ds_test = SlidingWindowDataset(test_s, cfg.seq_len, cfg.pred_horizon)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return dl_train, dl_val, dl_test, scaler


def run_experiment(csv_path: str, target_col: str, date_col: str = None, cfg: TrainConfig = CFG):
    print("Loading data...")
    series = load_series_from_csv(csv_path, target_col, date_col)
    print(f"Total observations: {len(series)}")

    dl_train, dl_val, dl_test, scaler = build_loaders(series, cfg)

    print("Building model...")
    model = GRQLSTMInformer(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0

    print("Start training...")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, optimizer, cfg)
        va_loss = evaluate(model, dl_val, cfg)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            patience = 0
            torch.save(model.state_dict(), cfg.save_path)
            improved = "*"
        else:
            patience += 1
            improved = ""

        print(f"Epoch {epoch:03d} | train_pinball={tr_loss:.6f} | val_pinball={va_loss:.6f} {improved}")

        if patience >= cfg.early_stop_patience:
            print("Early stopping.")
            break

    print("Loading best model for testing...")
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))

    print("Testing...")
    det, qrisk, ws = test_model(model, dl_test, cfg, scaler)

    results = {"deterministic": det, "q_risk": qrisk, "winkler_score_95": ws}
    print(json.dumps(results, indent=2))
    return results, cfg.save_path


if __name__ == "__main__":
    DATA_CFG = {
        "csv_path": r"Path_of_the_csv_file.csv",
        "target_col": "target_column",
        "date_col": "date"
    }

    if not os.path.exists(DATA_CFG["csv_path"]):
        print(f"CSV not found: {DATA_CFG['csv_path']}")
    else:
        run_experiment(**DATA_CFG, cfg=CFG)
