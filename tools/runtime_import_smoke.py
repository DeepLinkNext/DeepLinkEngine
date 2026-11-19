"""Exercise the native vision path used by the frozen multimodal processor."""
import argparse
import json
from pathlib import Path
import traceback


def check():
    import numpy as np
    import cv2
    import torch
    import torchvision
    from transformers import AutoProcessor
    from torchvision.io import decode_png, encode_png
    image = np.arange(12 * 13 * 3, dtype=np.uint8).reshape(12, 13, 3)
    success, encoded = cv2.imencode(".png", image)
    if not success or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image):
        raise RuntimeError("OpenCV PNG round trip differs")
    pixels = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
    if not torch.equal(decode_png(encode_png(pixels)), pixels):
        raise RuntimeError("TorchVision PNG round trip differs")
    selected = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
                                   torch.tensor([0.9, 0.8]), 0.5)
    if selected.tolist() != [0]:
        raise RuntimeError("TorchVision CPU NMS result differs")
    return dict(verified=True, model_loaded=False, numpy=np.__version__,
                opencv=cv2.__version__, torch=torch.__version__,
                torchvision=torchvision.__version__, processor=AutoProcessor.__name__,
                checks=["opencv_png", "torchvision_png", "torchvision_nms", "auto_processor_import"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check()
    except Exception:
        result = dict(verified=False, error=traceback.format_exc(), model_loaded=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["verified"] else 1)
