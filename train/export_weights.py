"""Save half-precision copies of trained weights (best_fp16.pt) so they fit in a GitHub repository."""
import sys
from pathlib import Path
import torch
for r in sys.argv[1:]:
    run = Path(r)
    sd = torch.load(run / "best.pt", map_location="cpu")
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, run / "best_fp16.pt")
    print(f"{run.name}: {(run / 'best.pt').stat().st_size / 1e6:.0f} MB -> {(run / 'best_fp16.pt').stat().st_size / 1e6:.0f} MB")
