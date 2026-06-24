"""Run the PyTorch reference and dump per-stage golden fixtures for the Swift port.

Outputs into ../fixtures/:
  - input_image.png            the (downloaded) demo image
  - stage_tensors.safetensors  flat dict of all intermediate + final tensors (fp32)
  - postprocess.npz            final post-processed detections (variable-length)
  - meta.json                  shapes + scalar metadata
"""
import json, os
import numpy as np
import torch
import requests
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoImageProcessor, AutoModelForObjectDetection

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "..", "fixtures")
os.makedirs(FIX, exist_ok=True)

MP = "PaddlePaddle/PP-DocLayoutV3_safetensors"
IMG_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"

torch.manual_seed(0)
model = AutoModelForObjectDetection.from_pretrained(MP, dtype=torch.float32).eval()
ip = AutoImageProcessor.from_pretrained(MP)

image = Image.open(requests.get(IMG_URL, stream=True).raw).convert("RGB")
image.save(os.path.join(FIX, "input_image.png"))
print("image size (W,H):", image.size)

inputs = ip(images=[image], return_tensors="pt")
pixel_values = inputs["pixel_values"]
print("pixel_values:", tuple(pixel_values.shape), pixel_values.dtype)

# ---- hooks to capture backbone + encoder internals ----
captured = {}

def bb_hook(mod, inp, out):
    # out is a list of (source, mask) tuples, one per returned stage
    captured["backbone"] = [s.detach().clone() for (s, m) in out]

def enc_hook(mod, inp, out):
    # PPDocLayoutV3HybridEncoderOutput: last_hidden_state (list of 3), mask_feat
    captured["enc_pan"] = [t.detach().clone() for t in out.last_hidden_state]
    captured["mask_feat"] = out.mask_feat.detach().clone()

h1 = model.model.backbone.register_forward_hook(bb_hook)
h2 = model.model.encoder.register_forward_hook(enc_hook)

with torch.no_grad():
    outputs = model(pixel_values)

h1.remove(); h2.remove()

# ---- assemble flat tensor dict ----
tensors = {}
tensors["pixel_values"] = pixel_values.contiguous()
for i, t in enumerate(captured["backbone"]):
    tensors[f"bb_feat_{i}"] = t.contiguous()
for i, t in enumerate(captured["enc_pan"]):
    tensors[f"enc_pan_{i}"] = t.contiguous()
tensors["mask_feat"] = captured["mask_feat"].contiguous()

# final ForObjectDetection outputs
for name in ["logits", "pred_boxes", "order_logits", "out_masks",
             "intermediate_logits", "intermediate_reference_points",
             "init_reference_points", "enc_topk_logits", "enc_topk_bboxes",
             "enc_outputs_class", "enc_outputs_coord_logits"]:
    v = getattr(outputs, name, None)
    if v is not None and torch.is_tensor(v):
        tensors[name] = v.detach().contiguous()

save_file({k: v.cpu().clone() for k, v in tensors.items()}, os.path.join(FIX, "stage_tensors.safetensors"))

# ---- post-process ----
results = ip.post_process_object_detection(outputs, threshold=0.5,
                                           target_sizes=[image.size[::-1]])[0]
np.savez(os.path.join(FIX, "postprocess.npz"),
         scores=results["scores"].numpy(),
         labels=results["labels"].numpy(),
         boxes=results["boxes"].numpy(),
         order_seq=results["order_seq"].numpy(),
         **{f"poly_{i}": np.asarray(p) for i, p in enumerate(results["polygon_points"])})

meta = {
    "image_size_wh": list(image.size),
    "num_detections": int(len(results["scores"])),
    "shapes": {k: list(v.shape) for k, v in tensors.items()},
    "first_5_scores": results["scores"][:5].tolist(),
    "first_5_labels": results["labels"][:5].tolist(),
    "first_5_boxes": results["boxes"][:5].tolist(),
    "id2label": model.config.id2label,
}
with open(os.path.join(FIX, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("=== fixture shapes ===")
for k, v in tensors.items():
    print(f"  {k}: {tuple(v.shape)}")
print("num detections:", meta["num_detections"])
print("first labels:", [model.config.id2label[l] for l in meta["first_5_labels"]])
print("first scores:", [round(s, 3) for s in meta["first_5_scores"]])
print("DONE")
