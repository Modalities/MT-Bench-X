# This file was modified and originally stemmed from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.

# This file was modified and originally stemmed from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.


def str_to_torch_dtype(dtype: str):
    import torch

    if dtype is None:
        return None
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unrecognized dtype: {dtype}")
