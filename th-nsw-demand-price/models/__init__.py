from typing import Optional

from .cnn_gru import CNNGRUForecaster
from .bilstm  import BiLSTMForecaster
from .tpa_bilstm import TPABiLSTMForecaster
from .am_bilstm  import AMBiLSTMForecaster

# 未來新增模型就 import 進來，例如：
# from .transformer import TransformerForecaster

from config import model_accepted_keys


def create_model(
    model_type: str,
    input_size: int,
    horizon: int,
    n_targets: int = 2,
    **kwargs
) -> Optional[object]:  # 返回 nn.Module，但 typing 上寫 object 比較彈性
    """
    模型工廠函數：根據 model_type 建立對應的模型實例

    Args:
        model_type:   "cnn_gru", "bilstm", "transformer" ...
        input_size:   特徵維度 (從 datamodule 取得)
        horizon:      預測長度
        n_targets:    同時預測幾個目標變數 (預設 2: demand + price)
        **kwargs:     傳給具體模型的超參數

    Returns:
        nn.Module 實例，已移動到正確 device 前（但尚未 .to(device)）

    Raises:
        ValueError: 如果 model_type 不存在
    """

    # 從 kwargs 取出必要的參數，給預設值
    horizon   = kwargs.pop("horizon", 48)     # 如果沒傳就用預設
    n_targets = kwargs.pop("n_targets", 2)

    model_type_lower = model_type.lower().strip()

    # 取得這個模型接受的鍵集合
    accepted_keys = model_accepted_keys.get(model_type_lower, set())

    if not accepted_keys:
        raise ValueError(
            f"不支援的 model_type: {model_type!r}\n"
            f"已知支援的類型：{list(model_accepted_keys.keys())}"
        )

    # 過濾出只屬於這個模型的參數
    model_kwargs = {
        k: v for k, v in kwargs.items()
        if k in accepted_keys
    }

    if model_type == "cnn_gru":
        return CNNGRUForecaster(
            input_size = input_size,
            horizon = horizon,
            n_targets = n_targets,
            **model_kwargs
        )
    elif model_type == "bilstm":
        return BiLSTMForecaster(
            input_size = input_size,
            horizon = horizon,
            n_targets = n_targets,
            **model_kwargs
        )
    elif model_type_lower == "tpa_bilstm":
        return TPABiLSTMForecaster(
            input_size = input_size, 
            horizon = horizon, 
            n_targets = n_targets, 
            **model_kwargs
        )
    elif model_type_lower == "am_bilstm":
        return AMBiLSTMForecaster(
            input_size = input_size, 
            horizon = horizon, 
            n_targets = n_targets, 
            **model_kwargs
        )


    # 未來新增模型只要在這裡加一條 elif 即可
    # elif model_type == "transformer":
    #     return TransformerForecaster(...)

    else:
        raise ValueError(
            f"不支援的 model_type: {model_type!r}\n"
        )