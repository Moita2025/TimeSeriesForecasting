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