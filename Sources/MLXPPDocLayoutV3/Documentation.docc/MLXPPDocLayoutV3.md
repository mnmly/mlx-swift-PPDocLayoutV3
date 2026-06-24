# ``MLXPPDocLayoutV3``

Document-layout detection for Apple Silicon — an MLX-Swift port of
PaddlePaddle's **PP-DocLayoutV3** (RT-DETR + HGNetV2-L, with mask and
reading-order heads).

## Overview

`MLXPPDocLayoutV3` runs the PP-DocLayoutV3 detector entirely on-device via
[mlx-swift](https://github.com/ml-explore/mlx-swift). Given a document image it
returns layout regions — each with a class label, confidence score, bounding
box, mask polygon, and **reading-order rank** — reproducing the official
`transformers` reference to detection-level parity (labels exact, scores ≤5e-3,
boxes ≤2px) at ~14× the throughput of the PyTorch CPU reference.

The high-level entry point is ``PPDocLayoutV3Pipeline``: point it at a directory
containing `config.json` + `model.safetensors` and call ``PPDocLayoutV3Pipeline/detect(imageURL:threshold:polygons:)``.

```swift
import MLXPPDocLayoutV3

let pipeline = try PPDocLayoutV3Pipeline(modelDirectory: modelDir)
for det in try pipeline.detect(imageURL: url) {
    print(pipeline.label(det.label), det.score, det.box)
}
```

For lower-level use, ``PPDocLayoutV3Model`` exposes the raw forward pass and
``PostProcess`` decodes its outputs.

## Topics

### Getting started

- ``PPDocLayoutV3Pipeline``
- ``ModelStore``

### Configuration

- ``PPDocLayoutV3Config``
- ``HGNetV2Config``

### Model

- ``PPDocLayoutV3Model``
- ``PPDocLayoutV3Output``

### Pre- and post-processing

- ``ImageProcessor``
- ``PostProcess``
- ``Detection``

### Weights

- ``loadWeights(url:into:dtype:verify:)``
