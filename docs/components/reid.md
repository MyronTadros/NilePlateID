# Vehicle Re-Identification (ReID)

Re-identify vehicles across cameras using deep learning embeddings.

## Overview

Vehicle ReID matches cars by visual appearance, independent of license plates. This is useful when plates are occluded or unreadable.

![ReID Architecture](../assets/reid_pipeline.png)

## How It Works

### 1. Registration

```
Upload image/video → Detect cars → OCR plates → Save crops to gallery
```

Cars are indexed by their plate ID in `data/gallery/{plate_id}/`.

### 2. Search

```
Query video → Detect cars → Extract features → Match with gallery → Return results
```

Each detected car is compared to the gallery using cosine similarity.

## Model Architecture

![Training Architecture](../assets/training_architecture.png)

### ResNet50-IBN Backbone

Instance-Batch Normalization combines:

- **Instance Norm**: Removes appearance variations (lighting, color)
- **Batch Norm**: Preserves discriminative features

```python
class IBN(nn.Module):
    def __init__(self, planes):
        self.IN = nn.InstanceNorm2d(planes // 2, affine=True)
        self.BN = nn.BatchNorm2d(planes // 2)
    
    def forward(self, x):
        split = x.chunk(2, 1)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        return torch.cat((out1, out2), 1)
```

### Loss Functions

#### Contrastive Loss

Pulls similar vehicles together, pushes different vehicles apart:

$$L_{contrastive} = \frac{1}{2N} \sum_{n=1}^{N} (y_n \cdot d_n^2 + (1-y_n) \cdot \max(0, m - d_n)^2)$$

Where:

- $d_n$ = Euclidean distance between embeddings
- $y_n$ = 1 if same vehicle, 0 otherwise
- $m$ = margin (typically 0.5-1.0)

#### Circle Loss

Provides better convergence with adaptive margins:

$$L_{circle} = \log \left[ 1 + \sum_{j} e^{\gamma \alpha_j^n (s_j^n - \Delta_n)} \cdot \sum_{k} e^{-\gamma \alpha_k^p (s_k^p - \Delta_p)} \right]$$

#### Combined Training

$$L_{total} = L_{ID} + \lambda_1 L_{contrastive} + \lambda_2 L_{circle}$$

## Usage

### Python API

```python
from src.reid.search import (
    _load_reid_model,
    _load_gallery_embeddings,
    _score_candidates,
    _filter_matches
)

# Load model
model = _load_reid_model(opts_path, checkpoint_path, device)

# Load gallery
gallery = _load_gallery_embeddings(
    gallery_dir, plate_id, model, input_size=224, device=device
)

# Score candidates
matches = _score_candidates(gallery, candidates, model, input_size, device)

# Filter by threshold
filtered = _filter_matches(matches, min_score=0.6, top_k=5)
```

### Streamlit App

1. Go to **🔍 Vehicle ReID** page
2. **Register**: Upload images/videos to build gallery
3. **Search**: Select a plate and upload query video
4. **Gallery**: View all registered vehicles

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 224 | ReID input image size |
| `batch_size` | 32 | Embedding batch size |
| `min_score` | 0.6 | Minimum cosine similarity |
| `top_k` | 5 | Max matches per frame |

## Limitations

- **Similar Vehicles**: Same model/color cars can be confused
- **Lighting**: Performance drops in low light
- **Angle**: Large viewpoint changes reduce accuracy
- **Occlusion**: Partially visible vehicles harder to match

## References

- **Paper**: Zheng et al., "Joint Discriminative and Generative Learning for Person Re-identification", CVPR 2019
- **Code**: [layumi/Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
- **Tutorial**: [Kaggle Vehicle ReID](https://www.kaggle.com/code/sosperec/vehicle-reid-tutorial/)
