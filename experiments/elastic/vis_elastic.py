import os
import sys
import torch

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from training.trainer     import Trainer
from visualization.fields import make_grid_2d, eval_velocity_norm, plot_field

CHECKPOINT = "checkpoints/elastic/ckpt_5000.pt"
OUTPUT_PNG = "images/elastic_vnorm.png"

X_RANGE = (0.0, 30.0)
Y_RANGE = (0.0, 30.0)
T_PLOT  = 0.1
F0      = 1400 / (500 * 0.15)
N       = 800

os.makedirs("images", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Device: {device}, step: {step}")

coords, _, _ = make_grid_2d(X_RANGE, Y_RANGE, N, t=T_PLOT, device=device)
mu = torch.tensor([[F0]], dtype=torch.float32, device=device)
vnorm = eval_velocity_norm(net, coords, mu, N)

plot_field(
    vnorm, X_RANGE, Y_RANGE,
    title=rf"$|\mathbf{{v}}(x,y,\,t={T_PLOT:g})|$",
    label=r"$|\mathbf{v}|$",
    cmap="magma",
    output_path=OUTPUT_PNG,
)
print(f"Saved -> {OUTPUT_PNG}")
