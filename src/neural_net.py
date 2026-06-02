from dataclasses import dataclass, asdict
import io
import logging
import os
import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import zstandard as zstd
import alphazero
from tracy_utils import tracy_zone

def get_storage_dtype():
    """Half dtype for storage. float16 is 8x more precise for [-1,1] data."""
    return torch.float16


def to_half_safe(tensor, dtype):
    """Convert tensor to half dtype, falling back to float32 if values overflow.

    Non-floating-point tensors (e.g. BatchNorm num_batches_tracked) are
    returned unchanged.
    """
    if not tensor.is_floating_point():
        return tensor
    if tensor.numel() == 0:
        return tensor.to(dtype)
    info = torch.finfo(dtype)
    max_val = tensor.abs().max().item()
    if max_val > info.max:
        logging.warning(
            f"Tensor max {max_val:.1f} exceeds {dtype} range ({info.max}), keeping float32"
        )
        return tensor
    return tensor.to(dtype)

# This is an autotuner for network speed.
torch.backends.cudnn.benchmark = True

@dataclass
class NNArgs:
    num_channels: int
    depth: int
    kernel_size: int
    dense_net: bool = False
    lr: float = 0.01
    cv: float = 1.5
    star_gambit_spatial: bool = False
    head_channels: int = 32
    head_pool: bool = True
    v_fc_hidden: int = -1    # -1 = auto-derive as head_channels * 8
    pi_fc_hidden: int = -1   # -1 = auto-derive as head_channels * 8
    v_head_convs: int = 0    # extra conv layers in value head (0 = current behavior)
    pi_head_convs: int = 0   # extra conv layers in policy head (0 = current behavior)
    v_fc_layers: int = 1     # hidden FC layers in value head (1 = current behavior)
    pi_fc_layers: int = 0    # hidden FC layers in policy head (0 = current single-projection)
    # "batch" (current default, nn.BatchNorm2d) or "layer" (GroupNorm(1, C) ==
    # LayerNorm across (C, H, W) per sample). LayerNorm avoids batch-statistic
    # propagation through bootstrap targets, a documented driver of feature
    # collapse / plasticity loss in deep RL (Liu et al. 2024).
    trunk_norm: str = "batch"
    # AlphaZero standard. Mirrors TrainConfig.weight_decay default. Old value
    # was 1e-3; star_gambit train-fixes validated 1e-4 empirically.
    weight_decay: float = 1e-4
    # Trunk activation. "relu" = nn.ReLU (default). "crelu" = Concatenated ReLU,
    # which preserves negative-direction features (doubles conv in-channels in
    # blocks). Abbas et al. 2023 found this the single most effective plasticity
    # intervention in continual RL benchmarks.
    trunk_act: str = "relu"
    # Orthogonal weight regularization on trunk conv weights. Loss term:
    #   orth_reg_lambda * sum_layers ||W_l^T W_l - I||_F^2  (per-out-channel-normalized).
    # 0 disables. Direct mechanism to keep effective rank high by penalizing
    # filter correlation. See Bjorck et al. 2021.
    orth_reg_lambda: float = 0.0

    def __post_init__(self):
        if self.v_fc_hidden == -1:
            self.v_fc_hidden = self.head_channels * 8
        if self.pi_fc_hidden == -1:
            self.pi_fc_hidden = self.head_channels * 8


def nnargs_from_config(config):
    """Construct NNArgs from a TrainConfig instance."""
    return NNArgs(
        num_channels=config.channels,
        depth=config.depth,
        kernel_size=config.kernel_size,
        dense_net=config.dense_net,
        lr=config.lr,
        cv=config.cv,
        star_gambit_spatial=config.star_gambit_spatial,
        head_channels=config.head_channels,
        head_pool=config.head_pool,
        v_head_convs=config.v_head_convs,
        pi_head_convs=config.pi_head_convs,
        v_fc_layers=config.v_fc_layers,
        pi_fc_layers=config.pi_fc_layers,
        trunk_norm=config.trunk_norm,
        weight_decay=config.weight_decay,
        trunk_act=config.trunk_act,
        orth_reg_lambda=config.orth_reg_lambda,
    )


def get_device():
    """Get the best available device for computation (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def conv(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding="same",
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 1)


def conv3x3(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 3)


def _make_norm(channels, norm_type):
    """Norm layer factory for conv feature maps.

    "batch" -> nn.BatchNorm2d (default, propagates batch stats; can drive
                feature collapse in bootstrapped RL).
    "layer" -> nn.GroupNorm(1, C). This is per-sample LayerNorm across
                (C, H, W) — what "LayerNorm for conv nets" conventionally
                means. Does NOT propagate batch statistics, so robust to the
                non-stationary target distribution in bootstrapped training.
    """
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unknown trunk_norm: {norm_type!r} (expected 'batch' or 'layer')")


class CReLU(nn.Module):
    """Concatenated ReLU: out = [ReLU(x), ReLU(-x)] along channel dim.

    Doubles channel count. From Shang et al. 2016; explicitly recommended
    by Abbas et al. 2023 ("Loss of Plasticity in Continual Deep RL") as the
    single most effective plasticity-preserving intervention they tested.
    Preserves negative-direction information that vanilla ReLU discards,
    eliminating the "dying ReLU" failure mode that drives effective-rank
    collapse in bootstrapped RL.
    """
    def forward(self, x):
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)


def _make_act(channels, act_type):
    """Activation factory. Returns (activation_module, out_channels_after_act).

    "relu"  -> nn.ReLU, no channel change.
    "crelu" -> CReLU, channels doubled. Caller must adjust downstream layer
               in-channels by 2x.
    """
    if act_type == "relu":
        return nn.ReLU(inplace=True), channels
    if act_type == "crelu":
        return CReLU(), 2 * channels
    raise ValueError(f"Unknown act_type: {act_type!r} (expected 'relu' or 'crelu')")


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, kernel_size=3,
                 norm_type="batch", act_type="relu"):
        super(DenseBlock, self).__init__()
        self.bn1 = _make_norm(in_channels, norm_type)
        self.relu1, post1_c = _make_act(in_channels, act_type)
        self.conv1 = conv1x1(post1_c, growth_rate * bn_size, 1)
        self.bn2 = _make_norm(growth_rate * bn_size, norm_type)
        self.relu2, post2_c = _make_act(growth_rate * bn_size, act_type)
        self.conv2 = conv(post2_c, growth_rate, kernel_size=kernel_size)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, kernel_size=3,
                 norm_type="batch", act_type="relu"):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = _make_norm(out_channels, norm_type)
        self.downsample = downsample
        self.bn1 = _make_norm(in_channels, norm_type)
        self.relu1, post1_c = _make_act(in_channels, act_type)
        self.conv1 = conv(post1_c, out_channels, stride, kernel_size=kernel_size)
        self.bn2 = _make_norm(out_channels, norm_type)
        self.relu2, post2_c = _make_act(out_channels, act_type)
        self.conv2 = conv(post2_c, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)
        out += residual
        return out


class NNArch(nn.Module):
    def __init__(self, game, args):
        super(NNArch, self).__init__()
        # game params
        in_channels, in_x, in_y = game.CANONICAL_SHAPE()
        self.dense_net = args.dense_net
        self.star_gambit_spatial = args.star_gambit_spatial
        self.head_pool = args.head_pool
        self.pi_fc_layers = args.pi_fc_layers
        HC = args.head_channels

        if args.pi_fc_layers > 0 and args.star_gambit_spatial:
            raise ValueError("pi_fc_layers not supported with star_gambit_spatial policy head")

        norm_type = getattr(args, "trunk_norm", "batch")
        act_type = getattr(args, "trunk_act", "relu")

        if not self.dense_net:
            self.conv1 = conv(
                in_channels, args.num_channels, kernel_size=args.kernel_size
            )
            self.bn1 = _make_norm(args.num_channels, norm_type)

        self.layers = []
        for i in range(args.depth):
            if self.dense_net:
                self.layers.append(
                    DenseBlock(
                        in_channels + args.num_channels * i,
                        args.num_channels,
                        kernel_size=args.kernel_size,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                self.layers.append(
                    ResidualBlock(
                        args.num_channels,
                        args.num_channels,
                        kernel_size=args.kernel_size,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.conv_layers = nn.Sequential(*self.layers)

        if self.dense_net:
            final_size = in_channels + args.num_channels * args.depth
            self.v_conv = conv1x1(final_size, HC)
            self.pi_conv = conv1x1(final_size, HC)
        else:
            self.v_conv = conv1x1(args.num_channels, HC)
            self.pi_conv = conv1x1(args.num_channels, HC)

        self.v_bn = nn.BatchNorm2d(HC)
        self.v_relu = nn.ReLU(inplace=True)

        # Extra value head conv layers (cheap per-head backbone extensions)
        if args.v_head_convs > 0:
            v_conv_layers = []
            for _ in range(args.v_head_convs):
                v_conv_layers.extend([
                    conv(HC, HC, kernel_size=args.kernel_size),
                    nn.BatchNorm2d(HC),
                    nn.ReLU(inplace=True),
                ])
            self.v_extra_convs = nn.Sequential(*v_conv_layers)

        if args.head_pool:
            self.v_pool = nn.AdaptiveAvgPool2d(1)
            v_fc1_in = HC
        else:
            v_fc1_in = HC * in_x * in_y
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(v_fc1_in, args.v_fc_hidden)
        self.v_fc1_relu = nn.ReLU(inplace=True)

        # Extra value FC hidden layers
        if args.v_fc_layers > 1:
            v_fc_extra = []
            for _ in range(args.v_fc_layers - 1):
                v_fc_extra.extend([
                    nn.Linear(args.v_fc_hidden, args.v_fc_hidden),
                    nn.ReLU(inplace=True),
                ])
            self.v_fc_extra = nn.Sequential(*v_fc_extra)

        self.v_fc2 = nn.Linear(args.v_fc_hidden, game.NUM_PLAYERS() + 1)
        self.v_softmax = nn.LogSoftmax(1)

        self.pi_bn = nn.BatchNorm2d(HC)
        self.pi_relu = nn.ReLU(inplace=True)

        # Extra policy head conv layers
        if args.pi_head_convs > 0:
            pi_conv_layers = []
            for _ in range(args.pi_head_convs):
                pi_conv_layers.extend([
                    conv(HC, HC, kernel_size=args.kernel_size),
                    nn.BatchNorm2d(HC),
                    nn.ReLU(inplace=True),
                ])
            self.pi_extra_convs = nn.Sequential(*pi_conv_layers)

        if self.star_gambit_spatial:
            # Star Gambit spatial policy head:
            # - Spatial actions: BOARD_DIM x BOARD_DIM x 10 action types per position
            # - Global actions: 18 deploy + 1 end_turn = 19
            self.sg_board_dim = in_x  # 11 for Skirmish/Clash, 13 for Battle
            self.sg_spatial_actions = in_x * in_y * 10
            self.sg_global_actions = 19  # deploy (18) + end_turn (1)

            # Spatial policy: conv output (B, 10, H, W) -> (B, H*W*10)
            self.pi_conv2 = conv1x1(HC, 10)
            self.pi_bn2 = nn.BatchNorm2d(10)

            # Global policy: for deploy and end_turn actions
            self.pi_flatten = nn.Flatten()
            if args.head_pool:
                self.pi_pool = nn.AdaptiveAvgPool2d(1)
                pi_global_in = HC
            else:
                pi_global_in = HC * in_x * in_y
            self.pi_global = nn.Sequential(
                nn.Linear(pi_global_in, args.pi_fc_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(args.pi_fc_hidden, self.sg_global_actions),
                nn.LayerNorm(self.sg_global_actions)
            )
            self.pi_softmax = nn.LogSoftmax(1)
        else:
            # Standard fully-connected policy head
            self.pi_flatten = nn.Flatten()
            if args.pi_fc_layers > 0:
                # Multi-layer policy FC head
                self.pi_fc1 = nn.Linear(in_x * in_y * HC, args.pi_fc_hidden)
                if args.pi_fc_layers > 1:
                    pi_fc_extra = []
                    for _ in range(args.pi_fc_layers - 1):
                        pi_fc_extra.extend([
                            nn.Linear(args.pi_fc_hidden, args.pi_fc_hidden),
                            nn.ReLU(inplace=True),
                        ])
                    self.pi_fc_extra = nn.Sequential(*pi_fc_extra)
                self.pi_fc_out = nn.Linear(args.pi_fc_hidden, game.NUM_MOVES())
            else:
                # Single projection (current default)
                self.pi_fc1 = nn.Linear(in_x * in_y * HC, game.NUM_MOVES())
            self.pi_softmax = nn.LogSoftmax(1)

    def forward(self, s):
        # s: batch_size x num_channels x board_x x board_y
        if not self.dense_net:
            s = self.conv1(s)
            s = self.bn1(s)
        s = self.conv_layers(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = self.v_relu(v)
        if hasattr(self, 'v_extra_convs'):
            v = self.v_extra_convs(v)
        if self.head_pool:
            v = self.v_pool(v)
        v = self.v_flatten(v)
        v = self.v_fc1(v)
        v = self.v_fc1_relu(v)
        if hasattr(self, 'v_fc_extra'):
            v = self.v_fc_extra(v)
        v = self.v_fc2(v)
        v = self.v_softmax(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = self.pi_relu(pi)
        if hasattr(self, 'pi_extra_convs'):
            pi = self.pi_extra_convs(pi)

        if self.star_gambit_spatial:
            # Spatial policy head for Star Gambit
            # s_flat for global actions, pi for spatial
            if self.head_pool:
                s_flat = self.pi_flatten(self.pi_pool(pi))
            else:
                s_flat = self.pi_flatten(pi)

            # Spatial actions: (B, HC, H, W) -> (B, 10, H, W) -> (B, H, W, 10) -> (B, H*W*10)
            pi_spatial = self.pi_conv2(pi)
            pi_spatial = self.pi_bn2(pi_spatial)
            pi_spatial = pi_spatial.permute(0, 2, 3, 1)  # (B, H, W, 10)
            batch_size = pi_spatial.shape[0]
            pi_spatial = pi_spatial.reshape(batch_size, -1)  # (B, H*W*10)

            # Global actions: deploy + end_turn
            pi_global = self.pi_global(s_flat)  # (B, 19)

            # Concatenate: spatial + global
            pi = torch.cat([pi_spatial, pi_global], dim=1)  # (B, H*W*10 + 19)
            pi = self.pi_softmax(pi)
        elif self.pi_fc_layers > 0:
            pi = self.pi_flatten(pi)
            pi = self.pi_fc1(pi)
            pi = torch.relu(pi)
            if hasattr(self, 'pi_fc_extra'):
                pi = self.pi_fc_extra(pi)
            pi = self.pi_fc_out(pi)
            pi = self.pi_softmax(pi)
        else:
            pi = self.pi_flatten(pi)
            pi = self.pi_fc1(pi)
            pi = self.pi_softmax(pi)

        return v, pi


class _CUDAGraphInference:
    """Cache CUDA graphs per batch-size bucket for fast inference."""

    def __init__(self, model, input_shape, device, dtype):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.dtype = dtype
        self._cache = {}  # bucket_size -> (graph, static_input, static_v, static_pi)

    def __call__(self, batch):
        bs = batch.shape[0]
        bucket = 1
        while bucket < bs:
            bucket *= 2
        if bucket not in self._cache:
            self._record(bucket)
        graph, static_input, static_v, static_pi = self._cache[bucket]
        static_input[:bs].copy_(batch)
        graph.replay()
        return static_v[:bs].clone(), static_pi[:bs].clone()

    def _forward(self, x):
        v, pi = self.model(x)
        return torch.exp(v).float(), torch.exp(pi).float()

    def _record(self, bucket_size):
        static_input = torch.zeros(
            bucket_size, *self.input_shape, device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            for _ in range(3):
                self._forward(static_input)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            static_v, static_pi = self._forward(static_input)
        torch.cuda.synchronize()
        self._cache[bucket_size] = (g, static_input, static_v, static_pi)

    def warmup(self, max_batch_size):
        """Pre-record graphs for all power-of-2 buckets."""
        max_bucket = 1
        while max_bucket < max_batch_size:
            max_bucket *= 2
        bucket = 1
        while bucket <= max_bucket:
            if bucket not in self._cache:
                self._record(bucket)
            bucket *= 2


class NNWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = NNArch(game, args)
        # Reference to the original eager module. enable_inference_optimizations()
        # may replace self.nnet with a JIT-traced RecursiveScriptModule (Pascal
        # path) or a torch.compile wrapper, neither of which exposes the eager
        # submodules (e.g. conv_layers) needed for hooking/regularization.
        self._eager_nnet = self.nnet
        self.weight_decay = getattr(args, "weight_decay", 1e-4)
        self.orth_reg_lambda = getattr(args, "orth_reg_lambda", 0.0)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9,
            weight_decay=self.weight_decay,
        )
        self.device = get_device()
        self.cv = args.cv
        self.nnet.to(self.device)
        self._amp_enabled = False
        self._graph_inference = None
        # non_blocking transfers are only safe on CUDA with pinned memory;
        # on MPS, intermediate tensors from .float().contiguous() can be GC'd
        # before the async transfer completes, causing data corruption.
        self._non_blocking = self.device.type == 'cuda'

    def enable_inference_optimizations(self, amp=True, compile=True):
        """Optimize model for inference on GPU. No-op on CPU."""
        is_gpu = self.device.type in ('cuda', 'mps')
        if amp and is_gpu:
            self._amp_enabled = True
            self.nnet.eval()
        if compile and is_gpu:
            try:
                if self.device.type == 'cuda':
                    cap = torch.cuda.get_device_capability(self.device)
                    if cap[0] < 7:
                        # Triton requires CC >= 7.0; manual JIT + CUDA graphs instead
                        logging.getLogger(__name__).info(
                            "CUDA capability %d.%d < 7.0: using JIT trace + CUDA graphs",
                            *cap,
                        )
                        self._apply_jit_and_cuda_graphs()
                        return
                self.nnet = torch.compile(self.nnet, mode="reduce-overhead")
            except Exception:
                pass  # graceful fallback

    def _apply_jit_and_cuda_graphs(self):
        """JIT trace + CUDA graph caching for older GPUs (CC < 7.0)."""
        try:
            in_shape = self.game.CANONICAL_SHAPE()
            dummy = torch.randn(1, *in_shape, device=self.device, dtype=torch.float32)
            self.nnet.eval()
            with torch.no_grad():
                traced = torch.jit.trace(self.nnet, dummy)
                traced = torch.jit.freeze(traced)
                traced(dummy)  # trigger optimizations
            self.nnet = traced
            self._graph_inference = _CUDAGraphInference(
                self.nnet, in_shape, self.device, torch.float32
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "JIT trace + CUDA graphs failed, falling back to eager: %s", e
            )
            self._graph_inference = None

    def warmup_graphs(self, max_batch_size):
        """Pre-record CUDA graphs for all bucket sizes up to max_batch_size."""
        if self._graph_inference is not None:
            self._graph_inference.warmup(max_batch_size)

    def set_lr(self, lr):
        """Set learning rate on all optimizer parameter groups."""
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    @tracy_zone
    def losses(self, dataset):
        self.nnet.eval()
        losses_v = []
        losses_pi = []
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
                canonical, target_vs, target_pis = batch
                canonical = canonical.float().contiguous().to(self.device, non_blocking=self._non_blocking)
                target_vs = target_vs.float().contiguous().to(self.device, non_blocking=self._non_blocking)
                target_pis = target_pis.float().contiguous().to(self.device, non_blocking=self._non_blocking)

                out_v, out_pi = self.nnet(canonical)
                losses_v.append(self.loss_v(target_vs, out_v))
                losses_pi.append(self.loss_pi(target_pis, out_pi))
        n = len(dataset)
        l_v = torch.stack(losses_v).sum().item() / n
        l_pi = torch.stack(losses_pi).sum().item() / n
        return l_v, l_pi

    @tracy_zone
    def sample_loss(self, dataset, size):
        self.nnet.eval()
        losses_gpu = []
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
                canonical, target_vs, target_pis = batch
                canonical = canonical.float().contiguous().to(self.device, non_blocking=self._non_blocking)
                target_pis = target_pis.float().contiguous().to(self.device, non_blocking=self._non_blocking)

                out_v, out_pi = self.nnet(canonical)
                l_pi = self.sample_loss_pi(target_pis, out_pi)
                losses_gpu.append(l_pi)
        return torch.cat(losses_gpu).cpu().numpy()

    @tracy_zone
    def train(self, batches, steps_to_train, run, epoch, total_train_steps, ema_averaging=True):
        self.nnet.train()

        v_loss = 0
        pi_loss = 0
        current_step = 0
        pbar = tqdm(
            total=steps_to_train, unit="batches", desc="Training NN", leave=False
        )
        past_states = []
        # Compute the 3 intermediate snapshot points (at 25%, 50%, 75% of training)
        snapshot_interval = steps_to_train // 4
        snapshot_steps = set()
        if ema_averaging and snapshot_interval > 0:
            snapshot_steps = {snapshot_interval, 2 * snapshot_interval, 3 * snapshot_interval}

        while current_step < steps_to_train:
            for batch in batches:
                if current_step in snapshot_steps:
                    past_states.append({k: v.data.clone() for k, v in self.nnet.named_parameters()})
                if current_step == steps_to_train:
                    break
                canonical, target_vs, target_pis = batch
                canonical = canonical.float().contiguous().to(self.device, non_blocking=self._non_blocking)
                target_vs = target_vs.float().contiguous().to(self.device, non_blocking=self._non_blocking)
                target_pis = target_pis.float().contiguous().to(self.device, non_blocking=self._non_blocking)

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v, out_pi = self.nnet(canonical)
                l_v = self.loss_v(target_vs, out_v)
                l_pi = self.loss_pi(target_pis, out_pi)
                total_loss = l_pi + l_v
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), float('inf'))
                self.optimizer.step()

                v_val = l_v.item()
                pi_val = l_pi.item()
                grad_norm_val = grad_norm.item()

                with torch.no_grad():
                    mask = target_pis > 0
                    target_entropy_val = -(target_pis[mask] * torch.log(target_pis[mask])).sum().item() / target_pis.size(0)
                kl_gap_val = pi_val - target_entropy_val

                run.track(
                    v_val,
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "value"},
                )
                run.track(
                    pi_val,
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "policy"},
                )
                run.track(
                    v_val + pi_val,
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "total"},
                )
                run.track(
                    target_entropy_val,
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "target_entropy"},
                )
                run.track(
                    kl_gap_val,
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "kl_gap"},
                )
                run.track(
                    grad_norm_val,
                    name="grad_norm",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                )

                # record loss and update progress bar.
                pi_loss += pi_val
                v_loss += v_val
                current_step += 1
                pbar.set_postfix(
                    {
                        "v loss": v_loss / current_step,
                        "pi loss": pi_loss / current_step,
                        "total": (v_loss + pi_loss) / current_step,
                    }
                )
                pbar.update()

        # Perform exponential averaging of network weights.
        if ema_averaging and past_states:
            past_states.append({k: v.data.clone() for k, v in self.nnet.named_parameters()})
            merged_states = past_states[0]
            for state in past_states[1:]:
                for k in merged_states.keys():
                    merged_states[k] = merged_states[k] * 0.75 + state[k] * 0.25
            nnet_dict = self.nnet.state_dict()
            nnet_dict.update(merged_states)
            self.nnet.load_state_dict(nnet_dict)

        pbar.close()
        return v_loss / steps_to_train, pi_loss / steps_to_train

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    @tracy_zone
    def process(self, batch):
        if self._graph_inference is not None:
            # copy_() into static_input handles dtype conversion implicitly
            res = self._graph_inference(batch.contiguous())
            if alphazero.tracy_is_enabled() and self.device.type == 'cuda':
                torch.cuda.synchronize()
            return res
        batch = batch.contiguous().to(self.device, non_blocking=self._non_blocking)
        self.nnet.eval()
        with torch.no_grad():
            if self._amp_enabled:
                with torch.amp.autocast(self.device.type, dtype=torch.float16):
                    v, pi = self.nnet(batch)
            else:
                v, pi = self.nnet(batch)
            res = (torch.exp(v).float(), torch.exp(pi).float())
        # GPU sync for accurate Tracy timing when profiling is enabled
        if alphazero.tracy_is_enabled():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()
        return res

    @tracy_zone
    def effective_rank(self, batch):
        """Participation ratio of the trunk's penultimate-layer activations.

        Measures effective dimensionality of the network's representation
        on a batch of inputs. Low values mean the representation has
        collapsed into fewer effective dimensions than the layer has
        channels — a classic capacity-binding signal in bootstrapped RL
        (Kumar et al. 2021, "Implicit Under-Parameterization in Deep RL").

        Hooks the output of `nnet.conv_layers` (the shared trunk feeding
        both heads) via a temporary forward hook, so regular forward/predict
        paths are untouched.

        Args:
            batch: tensor of canonical-form positions, shape [N, C, H, W].
                CPU or any device — moved internally.

        Returns:
            participation ratio in [1, channels], or nan on empty input.
        """
        if batch is None or len(batch) == 0:
            return float('nan')
        captured = []
        def _hook(_module, _inputs, outputs):
            captured.append(outputs.detach())
        # Use the eager module: self.nnet may be a JIT-traced/compiled module
        # without a conv_layers attribute, and a hook on the eager submodule
        # would not fire when calling the traced module anyway.
        net = self._eager_nnet
        net.eval()
        handle = net.conv_layers.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                x = batch.to(self.device, non_blocking=self._non_blocking).float()
                net(x)
        finally:
            handle.remove()
        if not captured:
            return float('nan')
        feats = captured[0].float()           # [B, C, H, W]
        feats = feats.mean(dim=(2, 3))        # spatial mean-pool -> [B, C]
        feats = feats - feats.mean(dim=0, keepdim=True)  # center per-feature
        s = torch.linalg.svdvals(feats)
        s2 = s ** 2
        denom = (s2 * s2).sum()
        if denom.item() == 0:
            return 1.0
        return float(((s2.sum() ** 2) / denom).item())

    def sample_loss_pi(self, targets, outputs):
        return -1 * torch.sum(targets * outputs, axis=1)

    def sample_loss_v(self, targets, outputs):
        return -self.cv * torch.sum(targets * outputs, axis=1)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        # return torch.sum((targets - outputs) ** 2) / targets.size()[0]
        return -self.cv * torch.sum(targets * outputs) / targets.size()[0]

    def trunk_orth_reg(self):
        """Orthogonal regularization on trunk conv weights.

        Returns sum_layers ||W^T W - I||_F^2 / out_channels  over the trunk's
        Conv2d modules. The /out_channels normalization makes lambda
        roughly architecture-independent across different channel counts.

        Returns 0 (as a Python int) if orth_reg_lambda is 0, so the caller can
        skip the loss-term addition without zero-tensor graph overhead.
        """
        if self.orth_reg_lambda <= 0:
            return 0
        device = self.device
        total = torch.zeros(1, device=device)
        for m in self._eager_nnet.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                W = m.weight.reshape(m.out_channels, -1)  # [out_c, in_c * kh * kw]
                gram = W @ W.transpose(0, 1)  # [out_c, out_c]
                I = torch.eye(m.out_channels, device=W.device, dtype=W.dtype)
                total = total + (gram - I).pow(2).sum() / max(m.out_channels, 1)
        return total

    @tracy_zone
    def save_checkpoint(
        self, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt",
        zstd_level=1,
    ):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)

        args_dict = asdict(self.args)

        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "args": args_dict,
                "game": self.game,
                "version": "5.0",  # zstd-compressed, float32 weights, no scheduler
            },
            buffer,
        )
        with open(filepath, 'wb') as f:
            f.write(zstd.ZstdCompressor(level=zstd_level, threads=-1).compress(buffer.getvalue()))

    @staticmethod
    @tracy_zone
    def load_checkpoint(
        Game, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt"
    ):
        if folder != "":
            filepath = os.path.join(folder, filename)
        else:
            filepath = filename
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")

        # Register safe globals for any alphazero game-state class. The
        # checkpoint pickles the original training class, which may not be
        # ``Game`` (e.g. a unified checkpoint stored as the parent class is
        # loaded for a variant subclass at play time).
        safe = [
            getattr(alphazero, name)
            for name in dir(alphazero)
            if name.endswith("GS") and isinstance(getattr(alphazero, name), type)
        ]
        torch.serialization.add_safe_globals(safe)

        # Try zstd decompression first, fall back to legacy uncompressed
        device = get_device()
        with open(filepath, 'rb') as f:
            raw = f.read()
        try:
            data = zstd.ZstdDecompressor().decompress(raw)
            checkpoint = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
        except zstd.ZstdError:
            checkpoint = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)

        stored_game = checkpoint["game"]
        compatible = stored_game == Game or (
            isinstance(Game, type) and issubclass(Game, stored_game)
        )
        assert compatible, (
            f"Mismatching game type when loading model: got: {stored_game.__name__} want: {Game.__name__}"
        )

        # Reconstruct NNArgs from dict, dropping removed fields
        args_dict = checkpoint["args"]
        args_dict.pop("lr_milestone", None)

        # Legacy checkpoints: fill in new fields with old hardcoded values
        if "head_channels" not in args_dict:
            args_dict["head_channels"] = 32
            args_dict["head_pool"] = False
            args_dict["v_fc_hidden"] = 256   # was hardcoded 32 * 8
            args_dict["pi_fc_hidden"] = 64   # was hardcoded 32 * 2

        # Head conv and FC depth (defaults match original architecture)
        args_dict.setdefault("v_head_convs", 0)
        args_dict.setdefault("pi_head_convs", 0)
        args_dict.setdefault("v_fc_layers", 1)
        args_dict.setdefault("pi_fc_layers", 0)

        args = NNArgs(**args_dict)

        net = NNWrapper(checkpoint["game"], args)
        net.nnet.load_state_dict(checkpoint["state_dict"])
        net.optimizer.load_state_dict(checkpoint["opt_state"])
        return net


def bench_network():

    Game = alphazero.OpenTaflGS
    depth = 4
    channels = 12
    dense_net = True
    batch_size = 1024
    nnargs = NNArgs(num_channels=channels, depth=depth, dense_net=dense_net)

    nn = NNWrapper(Game, nnargs)

    cs = Game.CANONICAL_SHAPE()
    dummy_input = torch.randn(batch_size, cs[0], cs[1], cs[2], dtype=torch.float)
    device = get_device()
    dummy_input = dummy_input.contiguous().to(device, non_blocking=True)

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # Warm up.
    for _ in range(50):
        _ = nn.process(dummy_input)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = nn.process(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        latency = np.sum(timings) / repetitions
        print(f"Inference Time: {latency:0.3f} ms")

    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = nn.process(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    throughput = (repetitions * batch_size) / total_time
    print(f"Throughput: {throughput:0.3f} samples/s")

    # with torch.profiler.profile() as prof:
    #     _ = nn.process(dummy_input)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    bench_network()
