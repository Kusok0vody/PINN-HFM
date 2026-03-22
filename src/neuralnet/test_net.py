import torch
import torch.nn as nn

from net import Net
from activations import *

def test_shapes(net, X, mu):
    out = net(X, mu)
    print("=== Tensor shapes ===")
    print(f"  X:   {list(X.shape)}")
    print(f"  mu:  {list(mu.shape)}")
    print()
    for name, tensor in out.items():
        print(f"  out['{name}']: {list(tensor.shape)}")
    print()


def test_grad(net, X, mu):
    out  = net(X, mu)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    print("=== Gradients ===")
    for name, p in net.named_parameters():
        status = "OK" if p.grad is not None else "NO GRADIENT"
        print(f"  {name:50s}  {status}")
    print()


def test_film_init(net):
    if not net.use_film:
        print("=== FiLM disabled ===\n")
        return
    last  = net.film.net[-1]
    d_h   = last.out_features // 2
    gamma = last.bias[:d_h]
    beta  = last.bias[d_h:]
    print("=== FiLM initialisation ===")
    print(f"  gamma: min={gamma.min():.4f}  max={gamma.max():.4f}  (expected ≈ 1)")
    print(f"  beta:  min={beta.min():.4f}   max={beta.max():.4f}   (expected ≈ 0)")
    print()


def test_fourier(net, X):
    if not net.use_fourier:
        print("=== Fourier embedding disabled ===\n")
        return
    emb = net.fourier(X)
    print("=== Fourier embedding ===")
    print(f"  input:  {list(X.shape)}")
    print(f"  output: {list(emb.shape)}")
    print(f"  frequencies: min={net.fourier.freqs.min():.3f}  max={net.fourier.freqs.max():.3f}")
    print()


def draw_graph(net, X, mu):
    try:
        from torchviz import make_dot
        out = net(X, mu)
        dot = make_dot(out[list(out.keys())[0]], params=dict(net.named_parameters()))
        dot.graph_attr.update(dpi="300")
        dot.node_attr.update(fontsize="10")
        dot.render("net_graph", format="pdf", cleanup=True)
        print("=== Graph saved to net_graph.pdf ===\n")
    except ImportError:
        print("=== torchviz not installed: pip install torchviz ===\n")


if __name__ == "__main__":
    N, M   = 50, 8
    x_dim  = 3
    mu_dim = 4

    net = Net(
        x_dim=x_dim,
        mu_dim=mu_dim,
        dx=32,
        dmu=32,
        d_h=128,
        encoder_layers=2,
        trunk_layers=4,
        head_layers=2,
        activation=nn.Tanh,
        outputs_config={
            "px": {"activation": nn.Tanh},
            "py": {"activation": nn.Tanh},
            "c":  {"multi": True, "K": 4, "activation": Morlet},
        },
        use_film=True,
        use_fourier=True,
        n_freqs=16,
        omega_min=1.0,
        omega_max=64.0,
    )

    X  = torch.randn(N, x_dim)
    mu = torch.randn(M, mu_dim)

    test_fourier(net, X)
    test_shapes(net, X, mu)
    test_film_init(net)
    test_grad(net, X, mu)
    draw_graph(net, X, mu)
    
    print(net)
    print()