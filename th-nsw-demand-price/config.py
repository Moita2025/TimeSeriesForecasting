model_configs = {
    "cnn_gru": {
        "hidden_size":    128,
        "num_gru_layers": 2,
        "num_cnn_filters":64,
        "kernel_size":    3,
        "dropout":        0.15,
        "bidirectional":  False,
        "use_pool":       False,
    },
    "bilstm": {
        "hidden_size":    128,
        "num_layers":     2,
        "dropout":        0.20,
        "bidirectional":  True,
    },
    # 未來可加 "transformer": {...}, "tcn": {...} 等
    "tpa_bilstm": {
        "hidden_size":    128,
        "num_layers":     2,
        "dropout":        0.20,
        "seq_len":        336,
    },
    "am_bilstm": {
        "hidden_size":    128,
        "num_layers":     2,
        "dropout":        0.20,
    },
    "dnn": {
        "hidden_sizes": [256, 128, 64],
        "dropout":      0.20,
        "seq_len":        336,
        # 可選：若想強制不同於其他模型的 dropout，可在此覆寫
    },
    "itransformer": {
        "dim":                    512,
        "depth":                  4,      # 先用較小值測試，資源允許可調到 6~8
        "heads":                  8,
        "dim_head":               64,
        "num_tokens_per_variate": 1,
        # "dropout":                0.1,
        "seq_len":        336,
        # "use_reversible_instance_norm": False,  # 若 lucidrains 版本支援，可加
    },
}

# 自动生成 model_accepted_keys
model_accepted_keys = {}
for model_name, config in model_configs.items():
    # 提取当前模型的所有键
    accepted_keys = set(config.keys())
    # 添加 horizon 和 n_targets
    accepted_keys.update({"horizon", "n_targets"})
    # 将结果存入新字典
    model_accepted_keys[model_name] = accepted_keys