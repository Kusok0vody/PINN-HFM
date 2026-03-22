import math
import torch
import torch.nn as nn

from mlp import MLP, MultiMLP, _init_linear

class FourierEmbedding(nn.Module):
    """
    Fixed Fourier embedding for input coordinates.
    Non-trainable — expands input representation with high-frequency features.

    x → [sin(2pi w1 x), cos(2pi w1 x), ..., sin(2pi wn x), cos(2pi wn x)]

    Frequencies form a geometric progression: wi = w_min · r^i
    """
    def __init__(
        self,
        in_dim:    int,
        n_freqs:   int,
        omega_min: float = 1.0,
        omega_max: float = 64.0,
    ):
        super().__init__()

        freqs = torch.logspace(
            math.log10(omega_min),
            math.log10(omega_max),
            n_freqs,
        )
        self.register_buffer("freqs", freqs)

        self.in_dim  = in_dim
        self.out_dim = in_dim * 2 * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x.unsqueeze(-1) * self.freqs
        emb  = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return emb.flatten(-2)


class Net(nn.Module):
    """
    PINN approximator architecture.

    Schema:
        X  -> [Fourier] → encoder_x → z_x --->
                                            |--> trunk → [FiLM] → heads → outputs
        µ  ->            encoder_mu → z_µ --->

    Args:
        x_dim:              input coordinate dimensionality
        mu_dim:             physical parameter dimensionality µ
        dx:                 latent dimensionality of coordinate representation
        dmu:                latent dimensionality of parameter representation
        d_h:                hidden layer width for trunk and heads
        encoder_layers:     number of layers in encoders
        trunk_layers:       number of layers in trunk
        head_layers:        number of layers in heads
        activation:         default activation for all MLP blocks
        encoder_activation: activation for encoders (overrides default)
        trunk_activation:   activation for trunk (overrides default)
        head_activation:    activation for heads (overrides default)
        film_activation:    activation for FiLM generator (overrides default)
        outputs_config:     output variables configuration:
                            {
                              "u": {},
                              "c": {"multi": True, "K": 4},
                            }
        use_film:           enable FiLM modulation
        use_fourier:        enable Fourier embedding for coordinates
        n_freqs:            number of frequencies in Fourier embedding
        omega_min:          minimum frequency
        omega_max:          maximum frequency
    """

    def __init__(
        self,
        x_dim:              int,
        mu_dim:             int,
        dx:                 int,
        dmu:                int,
        d_h:                int,
        encoder_layers:     int,
        trunk_layers:       int,
        head_layers:        int,
        activation,
        outputs_config:     dict,
        encoder_activation  = None,
        trunk_activation    = None,
        head_activation     = None,
        film_activation     = None,
        use_film:           bool  = True,
        use_fourier:        bool  = True,
        n_freqs:            int   = 16,
        omega_min:          float = 1.0,
        omega_max:          float = 64.0,
    ):
        super().__init__()

        self.use_film    = use_film
        self.use_fourier = use_fourier

        act_encoder = encoder_activation or activation
        act_trunk   = trunk_activation   or activation
        act_head    = head_activation    or activation
        act_film    = film_activation    or activation

        if use_fourier:
            self.fourier = FourierEmbedding(x_dim, n_freqs, omega_min, omega_max)
            encoder_x_in = self.fourier.out_dim
        else:
            self.fourier = None
            encoder_x_in = x_dim

        self.encoder_x = MLP(
            in_dim=encoder_x_in,
            out_dim=dx,
            hidden_dim=dx,
            num_layers=encoder_layers,
            activation=act_encoder,
        )

        self.encoder_mu = MLP(
            in_dim=mu_dim,
            out_dim=dmu,
            hidden_dim=dmu,
            num_layers=encoder_layers,
            activation=act_encoder,
        )

        self.trunk = MLP(
            in_dim=dx + dmu,
            out_dim=d_h,
            hidden_dim=d_h,
            num_layers=trunk_layers,
            activation=act_trunk,
        )

        if use_film:
            self.film = MLP(
                in_dim=dmu,
                out_dim=2 * d_h,
                hidden_dim=d_h,
                num_layers=2,
                activation=act_film,
            )
        else:
            self.film = None

        self.outputs = nn.ModuleDict()

        for name, cfg in outputs_config.items():
            act_out = cfg.get("activation", act_head)
            
            if cfg.get("multi", False):
                self.outputs[name] = MultiMLP(
                    in_dim=d_h,
                    hidden_dim=d_h,
                    num_layers=head_layers,
                    activation=act_out,
                    K=cfg.get("K", 2),
                )
            else:
                self.outputs[name] = MLP(
                    in_dim=d_h,
                    out_dim=1,
                    hidden_dim=d_h,
                    num_layers=head_layers,
                    activation=act_out,
                )
        self._init_weights()

    def _init_weights(self):
        """Initialises all MLP blocks with activation-aware weight initialisation."""

        def init_mlp(mlp: MLP):
            activation = None
            for module in mlp.net:
                if isinstance(module, nn.Linear):
                    is_first = (activation is None)
                    act = activation if activation is not None else nn.Tanh
                    _init_linear(module, act, is_first=is_first)
                else:
                    activation = module

        for mlp in [self.encoder_x, self.encoder_mu, self.trunk]:
            init_mlp(mlp)

        if self.film is not None:
            init_mlp(self.film)
            last = self.film.net[-1]
            nn.init.zeros_(last.weight)
            d_h = last.out_features // 2
            nn.init.ones_(last.bias[:d_h])
            nn.init.zeros_(last.bias[d_h:])

        for module in self.outputs.values():
            if isinstance(module, MLP):
                init_mlp(module)
            elif isinstance(module, MultiMLP):
                for mlp in module.mlps:
                    init_mlp(mlp)

    def forward(self, X: torch.Tensor, mu: torch.Tensor) -> dict:
        """
        Args:
            X:   (N, x_dim)  — spatial-temporal coordinates
            mu:  (M, mu_dim) — physical parameters

        Returns:
            dict: name → tensor (N, M)
        """
        if self.use_fourier:
            X = self.fourier(X)

        z_x  = self.encoder_x(X)
        z_mu = self.encoder_mu(mu)

        zx = z_x.unsqueeze(1)
        zm = z_mu.unsqueeze(0)

        Z = torch.cat([
            zx.expand(-1, zm.size(1), -1),
            zm.expand(zx.size(0), -1, -1),
        ], dim=-1)

        H = self.trunk(Z)

        if self.use_film:
            film_out    = self.film(z_mu)
            gamma, beta = film_out.chunk(2, dim=-1)
            gamma       = gamma.unsqueeze(0)
            beta        = beta.unsqueeze(0)
            H           = gamma * H + beta

        return {
            name: module(H).squeeze(-1)
            for name, module in self.outputs.items()
        }

    def __repr__(self) -> str:
        lines = ["Net("]
        lines.append(f"  use_fourier = {self.use_fourier}")
        lines.append(f"  use_film    = {self.use_film}")
        lines.append("")

        total = 0
        for name, p in self.named_parameters():
            total += p.numel()
            lines.append(f"  {name:50s}  {str(list(p.shape)):25s}  {p.numel():>10,}")

        lines.append("")
        lines.append(f"  {'Total parameters':50s}  {'':25s}  {total:>10,}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  {'Trainable parameters':50s}  {'':25s}  {trainable:>10,}")
        lines.append(")")
        return "\n".join(lines)