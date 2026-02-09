import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

from kan.KANLayer import KANLayer
from kan.spline import coef2curve


class KANModel(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=16, output_dim=3, device='cpu'):
        super(KANModel, self).__init__()
        self.device = device
        self.l1 = KANLayer(
            in_dim=input_dim,
            out_dim=hidden_dim,
            device=device,
            sparse_init=False,
            noise_scale=0.5
        )
        self.l2 = KANLayer(
            in_dim=hidden_dim,
            out_dim=output_dim,
            device=device,
            sparse_init=False,
            noise_scale=0.5
        )

    def forward(self, x):
        x, _, _, _ = self.l1(x)
        x, _, _, _ = self.l2(x)
        return x


def print_active_splines(layer, threshold=1e-4, name='Layer'):
    print(f"\n🔍 活跃 spline 函数（{name}，|scale_sp × coef| > {threshold}）:")
    count = 0
    for i in range(layer.in_dim):
        for j in range(layer.out_dim):
            sp = layer.scale_sp[i, j].item()
            coef = layer.coef[i, j, :].detach()
            activity = torch.sum(torch.abs(sp * coef)).item()
            if activity > threshold:
                print(f" - Spline {i} → {j} : activity = {activity:.4e}")
                count += 1
    print(f"总共激活的 spline 函数数：{count} / {layer.in_dim * layer.out_dim}")


def visualize_spline(layer, input_dim_index, output_dim_index, x_range=(-1.5, 1.5), resolution=300, layer_name='layer'):
    device = layer.device
    coef = layer.coef[input_dim_index, output_dim_index]
    sp = layer.scale_sp[input_dim_index, output_dim_index]
    if torch.sum(torch.abs(sp * coef)).item() < 1e-4:
        return

    x = torch.linspace(x_range[0], x_range[1], steps=resolution).view(-1, 1).to(device)
    X_dummy = torch.zeros(resolution, layer.in_dim, device=device)
    X_dummy[:, input_dim_index] = x.squeeze()

    with torch.no_grad():
        y = coef2curve(X_dummy, layer.grid, layer.coef, layer.k, device=device)
        spline_y = y[:, input_dim_index, output_dim_index].cpu()

    plt.figure(figsize=(5, 3))
    plt.plot(x.cpu(), spline_y, label=f"Spline {input_dim_index}→{output_dim_index}")
    plt.xlabel("Input")
    plt.ylabel("Spline Output")
    plt.title(f"Spline Function: {input_dim_index} → {output_dim_index}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs('spline_plots', exist_ok=True)
    plt.savefig(f"spline_plots/spline_{layer_name}_{input_dim_index}_{output_dim_index}.png")
    plt.close()

    os.makedirs('spline_csv', exist_ok=True)
    df = pd.DataFrame({'x': x.cpu().numpy().flatten(), 'y': spline_y.numpy().flatten()})
    df.to_csv(f"spline_csv/spline_{layer_name}_{input_dim_index}_{output_dim_index}.csv", index=False)


def visualize_all_splines(layer, layer_name='layer'):
    print(f"\n📈 正在可视化 {layer_name} 中所有 spline 函数...")
    for i in range(layer.in_dim):
        for j in range(layer.out_dim):
            visualize_spline(layer, i, j, layer_name=layer_name)


def train_from_csv(csv_path='data/train_data.csv'):
    df = pd.read_csv(csv_path)

    input_cols = ['u1', 'u2', 'u3', 'i1', 'i2', 'i3']
    output_cols = ['i1_next', 'i2_next', 'i3_next']

    X = torch.tensor(df[input_cols].values, dtype=torch.float32)
    Y = torch.tensor(df[output_cols].values, dtype=torch.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KANModel(input_dim=7, hidden_dim=16, output_dim=3, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 使用多个步长进行训练（可变步长仿真）
    step_list = [0.005, 0.01, 0.02, 0.04]

    losses = []
    for epoch in range(1000):
        model.train()
        epoch_loss = 0.0
        for delta in step_list:
            delta_t = torch.full((X.shape[0], 1), delta)
            input_X = torch.cat([X, delta_t], dim=1).to(device)
            target_Y = Y.to(device)

            pred = model(input_X)
            loss = loss_fn(pred, target_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(step_list)
        losses.append(avg_loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("KAN Training Loss with Variable Step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print_active_splines(model.l1, name='l1')
    print_active_splines(model.l2, name='l2')
    visualize_all_splines(model.l1, layer_name='l1')
    visualize_all_splines(model.l2, layer_name='l2')


if __name__ == '__main__':
    train_from_csv('data/train_data.csv')
