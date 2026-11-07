"""Save half-precision copies of trained weights (best_fp16.pt) so they fit in a GitHub repository.

best.pt is ~100 MB (E3: 103 MB, above GitHub's 100 MB file limit); best_fp16.pt is half that. The prototype loads
best_fp16.pt when best.pt is absent and converts it back to float32, which changes predictions only in the
4th decimal of the probabilities.

    python3 export_weights.py runs/E3 runs/E1 runs/E5 runs/E2b
"""
import sys
from pathlib import Path

import torch

for r in sys.argv[1:]:
    run = Path(r)
    sd = torch.load(run / "best.pt", map_location="cpu")
    half = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(half, run / "best_fp16.pt")
    print(f"{run.name}: {(run / 'best.pt').stat().st_size / 1e6:.0f} MB -> "
          f"{(run / 'best_fp16.pt').stat().st_size / 1e6:.0f} MB")
