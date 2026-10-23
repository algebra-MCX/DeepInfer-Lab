import os

import pytest
import torch

MODEL = os.environ.get("MINIDEEPINFER_TEST_MODEL", "tiny-random/deepseek-v4.1-flash-1017")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
