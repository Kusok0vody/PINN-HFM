import math
import torch
import torch.nn as nn

from network.mlp import MLP, MultiMLP, _init_linear

class FourierEmbedding(nn.Module):
    """
    Fixed Fourier embedding for input coordinates.
    Non-trainable — expands input representation with high-frequency features.

    x → [sin(pi w1 x), cos(pi w1 x), ..., sin(pi wn x), cos(pi wn x)]

    Frequencies form a geometric progression: wi = w_min · r^i

    Convention: omega counts full periods across the *whole* input domain,
    which Net rescales to [-1, 1] before this layer sees it. Hence the pi
    rather than 2*pi: pi * omega * x completes omega periods over a width of 2.
    omega = 1 therefore means one oscillation across the domain, in every
    problem, whatever the physical units.

    The earlier convention used 2*pi*omega*x, i.e. omega periods per unit
    coordinate. That made the same omega mean wildly different things per
    problem (64 oscillations across t in [0,1], but 1920 across t in [0,30]),
    and integer omega aliased x with x+1 inside the rescaled domain.

    A caution that the convention does not remove: a PINN differentiates the
    network twice, and mode omega contributes (pi omega)^2 to a second
    derivative. Large omega_max therefore swamps the PDE residual with
    high-frequency noise at initialisation, regardless of scaling. Keep
    omega_max near the number of oscillations the solution actually has.
    """
    def __init__(
        self,
        in_dim:    int,
        n_freqs:   int,
        omega_min: float = 1.0,
        omega_max: float = 8.0,
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
        proj = math.pi * x.unsqueeze(-1) * self.freqs
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
        film_layers:        int,
        activation,
        outputs_config:     dict,
        encoder_activation  = None,
        trunk_activation    = None,
        head_activation     = None,
        film_activation     = None,
        use_film:           bool  = True,
        use_fourier:        bool  = True,
        magnitude:          bool  = False,
        n_freqs:            int   = 16,
        omega_min:          float = 1.0,
        omega_max:          float = 64.0,
    ):
        super().__init__()

        self.config = {
            "x_dim":              x_dim,
            "mu_dim":             mu_dim,
            "dx":                 dx,
            "dmu":                dmu,
            "d_h":                d_h,
            "encoder_layers":     encoder_layers,
            "trunk_layers":       trunk_layers,
            "head_layers":        head_layers,
            "film_layers":        film_layers,
            "activation":         activation,
            "encoder_activation": encoder_activation,
            "trunk_activation":   trunk_activation,
            "head_activation":    head_activation,
            "film_activation":    film_activation,
            "outputs_config":     outputs_config,
            "use_film":           use_film,
            "use_fourier":        use_fourier,
            "magnitude":          magnitude,
            "n_freqs":            n_freqs,
            "omega_min":          omega_min,
            "omega_max":          omega_max,
        }

        self.use_film    = use_film
        self.use_fourier = use_fourier

        # Affine input rescaling to [-1, 1], applied inside forward() so that
        # autograd carries the chain-rule factor and the physics keeps
        # differentiating with respect to genuine physical coordinates.
        # Defaults are the identity map, so an un-configured Net behaves exactly
        # as before; PINN fills these in from the sampler's bounding box.
        self.register_buffer("x_lo",  -torch.ones(x_dim))
        self.register_buffer("x_hi",   torch.ones(x_dim))
        self.register_buffer("mu_lo", -torch.ones(mu_dim))
        self.register_buffer("mu_hi",  torch.ones(mu_dim))
        # Parameters swept on a log scale are rescaled in log space: a
        # log-uniform draw mapped linearly would pile most of its mass into a
        # sliver of the input range.
        self.register_buffer("mu_log", torch.zeros(mu_dim, dtype=torch.bool))

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
                num_layers=film_layers,
                activation=act_film,
            )
        else:
            self.film = None

        # An MLP with num_layers=1 is a single Linear: the loop that emits
        # activations runs zero times. Any activation configured for such a
        # block is silently discarded, which is easy to miss when a head is
        # given a Morlet or Sine and quietly gets none.
        # film_layers is deliberately excluded: one layer is its useful setting,
        # measured better than deeper generators, so warning about it would fire
        # on every sensible run. The consequence — film_activation does nothing
        # at film_layers = 1 — is documented on the argument instead.
        degenerate = [
            name for name, n in (("encoder_layers", encoder_layers),
                                 ("trunk_layers",   trunk_layers),
                                 ("head_layers",    head_layers))
            if n < 2
        ]
        if degenerate:
            import warnings
            warnings.warn(
                f"{', '.join(degenerate)} = 1: these blocks are a bare Linear and "
                "their activation is unused. Set 2 or more if you meant to keep it.",
                stacklevel=2,
            )

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
        # Amplitude as its own coordinate: one scalar per output variable and
        # per parameter setting, multiplying the head.
        #
        # Without it the size of the field is a property of the shared weights,
        # and a sweep asks one trunk for solutions whose amplitudes differ by
        # orders of magnitude. Near a resonance of the Helmholtz annulus the
        # exact solution grows without bound while its neighbours in the sweep
        # stay at order one, and the only per-setting freedom is FiLM, which has
        # to buy a threefold field against twenty-nine settings that want it
        # left where it is. Measured on a trained run under the hard boundary
        # ansatz: at three of thirty-two settings the loss would fall by a fifth
        # to a third at three times the amplitude reached, with an interior
        # minimum — the objective was asking for a field the optimiser could not
        # deliver through shared weights.
        #
        # This does leave the amplitude expressible two ways, here or in the
        # trunk. That redundancy is harmless; what was not harmless was the
        # multiplicative gain this replaces, and the difference is worth being
        # precise about. There, every derivative of the field being proportional
        # to the field made shrinking the gain lower every residual at once, so
        # u = 0 was a fixed point of the whole objective and one run reached it
        # exactly. That argument needs every term to vanish with the field, and
        # it no longer holds: the data residual is (u - u*) / scale, which at
        # u = 0 is of order one rather than zero. The term that knows the true
        # amplitude is what makes an explicit amplitude safe to expose.
        #
        # Normalising the head by its rms over the batch would remove the
        # redundancy outright, and is deliberately not done: it would make the
        # output a function of the whole point set rather than of the point, and
        # autograd would differentiate the normaliser too, so every reported
        # derivative — the Laplacian this problem is made of — would be wrong.
        self.magnitude_names = tuple(outputs_config) if magnitude else ()
        self.magnitude = None
        if magnitude:
            # Built last, and with the random stream put back where it was:
            # nn.Linear draws at construction, and those draws would shift every
            # initialisation that follows. Turning this flag on at a fixed seed
            # then leaves every other weight bit-for-bit unchanged, so an A/B
            # run differs by the thing under test and nothing else. The draws
            # themselves are discarded — the layer is zeroed just below.
            rng = torch.random.get_rng_state()
            self.magnitude = nn.Linear(dmu, len(outputs_config))
            torch.random.set_rng_state(rng)

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

        # Zero weight and bias means log s = 0, so every setting starts at gain
        # exactly one and the network begins as the one it would have been
        # without this at all. The amplitude spread is then something training
        # produces, not something initialisation has to be undone from.
        if self.magnitude is not None:
            nn.init.zeros_(self.magnitude.weight)
            nn.init.zeros_(self.magnitude.bias)

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

    def set_input_bounds(self, x_lo=None, x_hi=None, mu_lo=None, mu_hi=None,
                         mu_log=None):
        """
        Install the affine map that sends each input axis onto [-1, 1].

        Un-normalised inputs are the hidden cause of several nominally free
        hyperparameters. SIREN initialisation in mlp._init_linear assumes inputs
        in [-1, 1]; the Fourier embedding measures frequency in cycles per unit
        coordinate, so the same omega_max means something different in every
        problem. Fixing the scale once here makes both meaningful.

        Axes whose range collapses to a point are left as the identity — a
        single parameter setting must not divide by zero.
        """
        def install(lo_name, hi_name, lo, hi):
            if lo is None or hi is None:
                return
            lo = torch.as_tensor(lo, dtype=torch.float32).flatten()
            hi = torch.as_tensor(hi, dtype=torch.float32).flatten()
            buf_lo, buf_hi = getattr(self, lo_name), getattr(self, hi_name)
            if lo.shape != buf_lo.shape:
                raise ValueError(
                    f"{lo_name}: expected {tuple(buf_lo.shape)}, got {tuple(lo.shape)}"
                )
            degenerate = (hi - lo).abs() < 1e-12
            lo = torch.where(degenerate, -torch.ones_like(lo), lo)
            hi = torch.where(degenerate,  torch.ones_like(hi), hi)
            buf_lo.copy_(lo.to(buf_lo.device))
            buf_hi.copy_(hi.to(buf_hi.device))

        if mu_log is not None:
            flags = torch.as_tensor(mu_log, dtype=torch.bool).flatten()
            if flags.shape != self.mu_log.shape:
                raise ValueError(
                    f"mu_log: expected {tuple(self.mu_log.shape)}, got {tuple(flags.shape)}"
                )
            self.mu_log.copy_(flags.to(self.mu_log.device))

        install("x_lo",  "x_hi",  x_lo,  x_hi)
        install("mu_lo", "mu_hi", mu_lo, mu_hi)

    @staticmethod
    def _rescale(v: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    def _rescale_mu(self, mu: torch.Tensor) -> torch.Tensor:
        """Rescale parameters, taking the logarithm of log-swept columns first."""
        if self.mu_log.any():
            mu = torch.where(self.mu_log, mu.clamp_min(1e-30).log(), mu)
        return self._rescale(mu, self.mu_lo, self.mu_hi)

    def _apply_magnitude(self, out: dict, z_mu: torch.Tensor) -> dict:
        """Multiply each field by its own per-setting gain exp(s(µ))."""
        if self.magnitude is None:
            return out

        log_s = self.magnitude(z_mu)                       # (M, n_outputs)
        for j, name in enumerate(self.magnitude_names):
            # (1, M) against (N, M): the gain varies with the setting and is
            # constant in the coordinate, so it passes through every spatial
            # derivative as a factor and changes no residual's structure.
            out[name] = torch.exp(log_s[:, j]).unsqueeze(0) * out[name]
        return out

    def forward(self, X: torch.Tensor, mu: torch.Tensor) -> dict:
        """
        Args:
            X:   (N, x_dim)  — spatial-temporal coordinates
            mu:  (M, mu_dim) — physical parameters

        Returns:
            dict: name → tensor (N, M)
        """
        X  = self._rescale(X, self.x_lo, self.x_hi)
        mu = self._rescale_mu(mu)

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

        return self._apply_magnitude(
            {name: module(H).squeeze(-1) for name, module in self.outputs.items()},
            z_mu,
        )

    def forward_paired(self, X: torch.Tensor, mu: torch.Tensor) -> dict:
        """
        Same network, coordinates already replicated per parameter setting.

        Args:
            X:   (N, M, x_dim) — coordinate for every (point, parameter) pair
            mu:  (M, mu_dim)

        Returns:
            dict: name -> (N, M), identical to forward()

        forward() feeds one coordinate row to all M settings, so f[n, m] depends
        on a single leaf x[n] and a backward pass sums over m. Recovering the M
        components then costs M passes. Here every pair owns its leaf, so one
        pass over f.sum() yields all of them.

        Only encoder_x pays for the replication — roughly a twentieth of the
        forward at the usual widths. encoder_mu still sees M rows and is
        broadcast, and the trunk already ran on N*M elements either way.
        """
        n, m, _ = X.shape
        Xf = X.reshape(n * m, -1)
        if self.use_fourier:
            Xf = self.fourier(self._rescale(Xf, self.x_lo, self.x_hi))
        else:
            Xf = self._rescale(Xf, self.x_lo, self.x_hi)

        z_x  = self.encoder_x(Xf).reshape(n, m, -1)
        z_mu = self.encoder_mu(self._rescale_mu(mu))

        Z = torch.cat([z_x, z_mu.unsqueeze(0).expand(n, -1, -1)], dim=-1)
        H = self.trunk(Z)

        if self.use_film:
            gamma, beta = self.film(z_mu).chunk(2, dim=-1)
            H = gamma.unsqueeze(0) * H + beta.unsqueeze(0)

        return self._apply_magnitude(
            {name: module(H).squeeze(-1) for name, module in self.outputs.items()},
            z_mu,
        )

    def __repr__(self) -> str:
        lines = ["Net("]
        lines.append(f"  use_fourier = {self.use_fourier}")
        lines.append(f"  use_film    = {self.use_film}")
        lines.append(f"  magnitude   = {self.magnitude is not None}")
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
    
    def serialize_config(self) -> dict:
        from network.activations import serialize_activation
        import copy

        cfg = copy.deepcopy(self.config)

        for key in ("activation", "encoder_activation", "trunk_activation",
                    "head_activation", "film_activation"):
            if cfg[key] is not None:
                cfg[key] = serialize_activation(cfg[key])

        for name, out_cfg in cfg["outputs_config"].items():
            if "activation" in out_cfg and out_cfg["activation"] is not None:
                out_cfg["activation"] = serialize_activation(out_cfg["activation"])

        return cfg

    @staticmethod
    def deserialize_config(cfg: dict) -> dict:
        from network.activations import deserialize_activation
        import copy

        cfg = copy.deepcopy(cfg)

        for key in ("activation", "encoder_activation", "trunk_activation",
                    "head_activation", "film_activation"):
            if cfg[key] is not None:
                cfg[key] = deserialize_activation(cfg[key])

        for name, out_cfg in cfg["outputs_config"].items():
            if "activation" in out_cfg and out_cfg["activation"] is not None:
                out_cfg["activation"] = deserialize_activation(out_cfg["activation"])

        # Fields added after some checkpoints were written. Each default is the
        # constant the network used before the field existed, so an old
        # checkpoint reconstructs the architecture it was actually trained with.
        cfg.setdefault("film_layers", 2)
        cfg.setdefault("magnitude", False)
        # Dropped after measurement: a multiplicative output gain never paid
        # for itself at these amplitudes, and it made u = 0 a fixed point of
        # every gradient, which is how one run reached exactly zero and stayed.
        # Old checkpoints still carry the key, so it is discarded rather than
        # passed to a constructor that no longer takes it.
        cfg.pop("output_scaling", None)

        return cfg

    RESCALE_BUFFERS = ("x_lo", "x_hi", "mu_lo", "mu_hi")

    @staticmethod
    def load_weights(net: "Net", state: dict) -> "Net":
        """
        Load a state dict, tolerating checkpoints written before input
        rescaling existed.

        Such a checkpoint was trained on raw coordinates, so its weights are
        only meaningful with the identity map. Any rescaling already installed
        on this Net is therefore reset rather than left in place — silently
        keeping it would feed the old weights inputs they never saw.
        """
        state = {k: v for k, v in state.items()
                 if not k.startswith(("log_gain.", "gain_mu."))}
        missing, unexpected = net.load_state_dict(state, strict=False)

        absent = set(missing) - set(Net.RESCALE_BUFFERS)
        if absent or unexpected:
            raise RuntimeError(
                f"checkpoint does not match this architecture — "
                f"missing {sorted(absent)}, unexpected {sorted(unexpected)}"
            )

        if missing:
            for name in Net.RESCALE_BUFFERS:
                getattr(net, name).fill_(-1.0 if name.endswith("_lo") else 1.0)
            print(
                "checkpoint predates input rescaling: reset to the identity map "
                "so the loaded weights see the coordinates they were trained on"
            )
        return net

    @staticmethod
    def from_checkpoint(path: str, device="cpu") -> "Net":
        """
        Восстанавливает Net полностью из чекпоинта — архитектура + веса.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = Net.deserialize_config(ckpt["net_config"])
        net  = Net(**cfg)
        Net.load_weights(net, ckpt["net"])
        net.to(device)
        return net