import torch, time, os
from transformers import AutoModelForObjectDetection
from safetensors.torch import load_file
MP="PaddlePaddle/PP-DocLayoutV3_safetensors"
m=AutoModelForObjectDetection.from_pretrained(MP,dtype=torch.float32).eval()
FIX=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","fixtures")
px=load_file(os.path.join(FIX,"stage_tensors.safetensors"))["pixel_values"]
with torch.no_grad():
    m(px)  # warmup
    N=15; t=time.time()
    for _ in range(N): m(px)
    dt=(time.time()-t)/N
print(f"Python (torch CPU) avg {dt:.3f}s / image  ({N} iters)")
