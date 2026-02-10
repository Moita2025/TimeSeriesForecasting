import torch.nn as nn
from iTransformer import iTransformer


class iTransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_targets: int = 2,
        dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        num_tokens_per_variate: int = 1,
        **kwargs
    ):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.num_variates = n_targets  # 只預測 demand + price

        lookback_len = kwargs.get("seq_len")
        if lookback_len is None:
            raise ValueError("seq_len must be provided in kwargs")

        self.itransformer = iTransformer(
            num_variates = self.num_variates,
            lookback_len = lookback_len,
            pred_length  = horizon,
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            num_tokens_per_variate = num_tokens_per_variate,
        )

        if input_size != self.num_variates:
            self.input_proj = nn.Linear(input_size, self.num_variates)
        else:
            self.input_proj = nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)          # 投影到只有 n_targets 維
        out = self.itransformer(x)
        return out