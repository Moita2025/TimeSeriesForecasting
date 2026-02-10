import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Optional
import time

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
    criterion: Optional[nn.Module] = None,
    optimizer_class = Adam,
    optimizer_kwargs: dict = None,
    scheduler = None,
    scheduler_kwargs: dict = None,
    verbose: bool = True,
    print_every: int = 5,
) -> nn.Module:
    """
    簡單但結構清晰的訓練函數，專注於時序預測任務。

    Args:
        model: 要訓練的模型
        train_loader, val_loader: 訓練與驗證的 DataLoader
        epochs: 最大訓練輪數
        lr: 初始學習率
        patience: early stopping 的耐心值
        grad_clip: 梯度裁剪的最大範圍（設為 0 則不裁剪）
        device: 訓練裝置
        criterion: 損失函數（預設 MSELoss）
        optimizer_class: 優化器類別（預設 Adam）
        optimizer_kwargs: 傳給優化器的額外參數
        scheduler: lr scheduler（可選）
        scheduler_kwargs: scheduler 的初始化參數
        verbose: 是否印出訓練過程
        print_every: 每多少個 epoch 印一次較詳細資訊

    Returns:
        訓練完成後的最佳模型（已載入最佳權重）
    """
    model = model.to(device)

    if criterion is None:
        criterion = nn.MSELoss()

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_kwargs)

    if scheduler is not None and scheduler_kwargs:
        scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_state_dict = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        # ── Training ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)               # (batch, horizon, n_targets)
            loss = criterion(pred, y_batch)

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_size = X_batch.size(0)
            train_loss += loss.item() * batch_size
            n_samples += batch_size

        train_loss /= n_samples

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                batch_size = X_batch.size(0)
                val_loss += loss.item() * batch_size
                n_val_samples += batch_size

        val_loss /= n_val_samples

        # ── Scheduler step ──────────────────────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ── Early stopping & best model ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Logging ─────────────────────────────────────────────────
        if verbose and (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            eta = elapsed * (epochs - epoch - 1) / (epoch + 1) if epoch > 0 else 0

            msg = (f"Epoch {(epoch+1):02d}/{epochs} | "
                   f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                   f"Best Val: {best_val_loss:.6f} | Patience: {patience_counter:02d}/{patience}")

            if scheduler is not None:
                current_lr = optimizer.param_groups[0]['lr']
                msg += f" | LR: {current_lr:.2e}"

            msg += f" | Elapsed: {elapsed:3.1f}s ETA: {eta:.0f}s"
            print(msg)

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 載入最佳權重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"已載入最佳模型 (val loss = {best_val_loss:.6f})")

    total_time = time.time() - start_time
    print(f"訓練完成，總耗時 {total_time:.1f} 秒")

    return model