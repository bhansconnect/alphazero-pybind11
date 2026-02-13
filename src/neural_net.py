from collections import namedtuple
import io
import logging
import os
import threading
import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import zstandard as zstd
import alphazero
from tracy_utils import tracy_zone

ZSTD_LEVEL = 3
HALF_DTYPE = torch.float16


def get_storage_dtype():
    """Best half dtype for storage: bfloat16 if hardware supports it, else float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
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

NNArgs = namedtuple(
    "NNArgs",
    ["num_channels", "depth", "kernel_size", "dense_net", "lr", "cv",
     "star_gambit_spatial"],
    defaults=(False, 0.01, 1.5, False),
)


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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_channels, growth_rate * bn_size, 1)
        self.bn2 = nn.BatchNorm2d(growth_rate * bn_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(growth_rate * bn_size, growth_rate, kernel_size=kernel_size)

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
    def __init__(self, in_channels, out_channels, downsample=False, kernel_size=3):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_channels, out_channels, stride, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size)

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

        if not self.dense_net:
            self.conv1 = conv(
                in_channels, args.num_channels, kernel_size=args.kernel_size
            )
            self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.layers = []
        for i in range(args.depth):
            if self.dense_net:
                self.layers.append(
                    DenseBlock(
                        in_channels + args.num_channels * i,
                        args.num_channels,
                        kernel_size=args.kernel_size,
                    )
                )
            else:
                self.layers.append(
                    ResidualBlock(
                        args.num_channels,
                        args.num_channels,
                        kernel_size=args.kernel_size,
                    )
                )
        self.conv_layers = nn.Sequential(*self.layers)

        if self.dense_net:
            final_size = in_channels + args.num_channels * args.depth
            self.v_conv = conv1x1(final_size, 32)
            self.pi_conv = conv1x1(final_size, 32)
        else:
            self.v_conv = conv1x1(args.num_channels, 32)
            self.pi_conv = conv1x1(args.num_channels, 32)

        self.v_bn = nn.BatchNorm2d(32)
        self.v_relu = nn.ReLU(inplace=True)
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(32 * in_x * in_y, 256)
        self.v_fc1_relu = nn.ReLU(inplace=True)
        self.v_fc2 = nn.Linear(256, game.NUM_PLAYERS() + 1)
        self.v_softmax = nn.LogSoftmax(1)

        self.pi_bn = nn.BatchNorm2d(32)
        self.pi_relu = nn.ReLU(inplace=True)

        if self.star_gambit_spatial:
            # Star Gambit spatial policy head:
            # - Spatial actions: BOARD_DIM x BOARD_DIM x 10 action types per position
            # - Global actions: 18 deploy + 1 end_turn = 19
            self.sg_board_dim = in_x  # 9 for Skirmish/Clash, 11 for Battle
            self.sg_spatial_actions = in_x * in_y * 10
            self.sg_global_actions = 19  # deploy (18) + end_turn (1)

            # Spatial policy: conv output (B, 10, H, W) -> (B, H*W*10)
            self.pi_conv2 = conv1x1(32, 10)
            self.pi_bn2 = nn.BatchNorm2d(10)

            # Global policy: for deploy and end_turn actions
            self.pi_flatten = nn.Flatten()
            self.pi_global = nn.Sequential(
                nn.Linear(32 * in_x * in_y, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.sg_global_actions),
                nn.LayerNorm(self.sg_global_actions)
            )
            self.pi_softmax = nn.LogSoftmax(1)
        else:
            # Standard fully-connected policy head
            self.pi_flatten = nn.Flatten()
            self.pi_fc1 = nn.Linear(in_x * in_y * 32, game.NUM_MOVES())
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
        v = self.v_flatten(v)
        v = self.v_fc1(v)
        v = self.v_fc1_relu(v)
        v = self.v_fc2(v)
        v = self.v_softmax(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = self.pi_relu(pi)

        if self.star_gambit_spatial:
            # Spatial policy head for Star Gambit
            # s_flat for global actions, pi for spatial
            s_flat = self.pi_flatten(pi)

            # Spatial actions: (B, 32, H, W) -> (B, 10, H, W) -> (B, H, W, 10) -> (B, H*W*10)
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
        else:
            pi = self.pi_flatten(pi)
            pi = self.pi_fc1(pi)
            pi = self.pi_softmax(pi)

        return v, pi


class _CUDAGraphInference:
    """Cache CUDA graphs per batch-size bucket for fast inference."""
    _record_lock = threading.Lock()

    def __init__(self, model, input_shape, device, use_autocast):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.use_autocast = use_autocast
        self._cache = {}  # bucket_size -> (graph, static_input, static_v, static_pi)
        self._lock = threading.Lock()

    def __call__(self, batch):
        bs = batch.shape[0]
        bucket = 1
        while bucket < bs:
            bucket *= 2
        if bucket not in self._cache:
            with _CUDAGraphInference._record_lock:
                if bucket not in self._cache:
                    self._record(bucket)
        with self._lock:
            graph, static_input, static_v, static_pi = self._cache[bucket]
            static_input[:bs].copy_(batch)
            graph.replay()
            return static_v[:bs].clone(), static_pi[:bs].clone()

    def _forward(self, x):
        if self.use_autocast:
            with torch.autocast(device_type='cuda', dtype=HALF_DTYPE):
                v, pi = self.model(x)
        else:
            v, pi = self.model(x)
        return torch.exp(v).float(), torch.exp(pi).float()

    def _record(self, bucket_size):
        static_input = torch.zeros(
            bucket_size, *self.input_shape, device=self.device
        )
        # Warmup (required before CUDA graph capture)
        with torch.no_grad():
            for _ in range(3):
                self._forward(static_input)
        # Record
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            static_v, static_pi = self._forward(static_input)
        self._cache[bucket_size] = (g, static_input, static_v, static_pi)


class NNWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = NNArch(game, args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
        )
        self.device = get_device()
        self.cv = args.cv
        self.nnet.to(self.device)
        self.use_autocast = False
        self._graph_inference = None

    def enable_inference_optimizations(self, autocast=True, compile=True):
        """Optimize model for inference on GPU. No-op on CPU."""
        is_gpu = self.device.type in ('cuda', 'mps')
        self.use_autocast = autocast and is_gpu
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
            dummy = torch.randn(1, *in_shape, device=self.device)
            self.nnet.eval()
            with torch.no_grad():
                traced = torch.jit.trace(self.nnet, dummy)
                traced = torch.jit.freeze(traced)
                traced(dummy)  # trigger optimizations
            self.nnet = traced
            self._graph_inference = _CUDAGraphInference(
                self.nnet, in_shape, self.device, self.use_autocast
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "JIT trace + CUDA graphs failed, falling back to eager: %s", e
            )
            self._graph_inference = None

    def set_lr(self, lr):
        """Set learning rate on all optimizer parameter groups."""
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    @tracy_zone
    def losses(self, dataset):
        self.nnet.eval()
        l_v = 0
        l_pi = 0
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
                canonical, target_vs, target_pis = batch
                canonical = canonical.contiguous().to(self.device, non_blocking=True)
                target_vs = target_vs.contiguous().to(self.device, non_blocking=True)
                target_pis = target_pis.contiguous().to(self.device, non_blocking=True)

                if self.use_autocast:
                    with torch.autocast(device_type=self.device.type, dtype=HALF_DTYPE):
                        out_v, out_pi = self.nnet(canonical)
                else:
                    out_v, out_pi = self.nnet(canonical)
                l_v += self.loss_v(target_vs, out_v).item()
                l_pi += self.loss_pi(target_pis, out_pi).item()
        return l_v / len(dataset), l_pi / len(dataset)

    @tracy_zone
    def sample_loss(self, dataset, size):
        loss = np.zeros(size)
        self.nnet.eval()
        i = 0
        with torch.no_grad():
            for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
                canonical, target_vs, target_pis = batch
                canonical = canonical.contiguous().to(self.device, non_blocking=True)
                target_vs = target_vs.contiguous().to(self.device, non_blocking=True)
                target_pis = target_pis.contiguous().to(self.device, non_blocking=True)

                if self.use_autocast:
                    with torch.autocast(device_type=self.device.type, dtype=HALF_DTYPE):
                        out_v, out_pi = self.nnet(canonical)
                else:
                    out_v, out_pi = self.nnet(canonical)
                l_v = self.sample_loss_v(target_vs, out_v)
                l_pi = self.sample_loss_pi(target_pis, out_pi)
                total_loss = l_pi + l_v
                batch_len = total_loss.shape[0]
                loss[i:i + batch_len] = total_loss.cpu().numpy()
                i += batch_len
        return loss

    @tracy_zone
    def train(self, batches, steps_to_train, run, epoch, total_train_steps):
        self.nnet.train()

        v_loss = 0
        pi_loss = 0
        current_step = 0
        pbar = tqdm(
            total=steps_to_train, unit="batches", desc="Training NN", leave=False
        )
        past_states = []
        while current_step < steps_to_train:
            for batch in batches:
                if (
                    steps_to_train // 4 > 0
                    and current_step % (steps_to_train // 4) == 0
                    and current_step != 0
                ):
                    # Snapshot model weights
                    past_states.append(dict(self.nnet.named_parameters()))
                if current_step == steps_to_train:
                    break
                canonical, target_vs, target_pis = batch
                canonical = canonical.contiguous().to(self.device, non_blocking=True)
                target_vs = target_vs.contiguous().to(self.device, non_blocking=True)
                target_pis = target_pis.contiguous().to(self.device, non_blocking=True)

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v, out_pi = self.nnet(canonical)
                l_v = self.loss_v(target_vs, out_v)
                l_pi = self.loss_pi(target_pis, out_pi)
                total_loss = l_pi + l_v
                total_loss.backward()
                self.optimizer.step()

                run.track(
                    l_v.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "value"},
                )
                run.track(
                    l_pi.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "policy"},
                )
                run.track(
                    l_v.item() + l_pi.item(),
                    name="loss",
                    epoch=epoch,
                    step=total_train_steps + current_step,
                    context={"type": "total"},
                )

                # record loss and update progress bar.
                pi_loss += l_pi.item()
                v_loss += l_v.item()
                current_step += 1
                pbar.set_postfix(
                    {
                        "v loss": v_loss / current_step,
                        "pi loss": pi_loss / current_step,
                        "total": (v_loss + pi_loss) / current_step,
                    }
                )
                pbar.update()

        # Perform expontential averaging of network weights.
        past_states.append(dict(self.nnet.named_parameters()))
        merged_states = past_states[0]
        for state in past_states[1:]:
            for k in merged_states.keys():
                merged_states[k].data = (
                    merged_states[k].data * 0.75 + state[k].data * 0.25
                )
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
        batch = batch.contiguous().to(self.device, non_blocking=True)
        if self._graph_inference is not None:
            res = self._graph_inference(batch)
            if alphazero.tracy_is_enabled() and self.device.type == 'cuda':
                torch.cuda.synchronize()
            return res
        self.nnet.eval()
        with torch.no_grad():
            if self.use_autocast:
                with torch.autocast(device_type=self.device.type, dtype=HALF_DTYPE):
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

    def sample_loss_pi(self, targets, outputs):
        return -1 * torch.sum(targets * outputs, axis=1)

    def sample_loss_v(self, targets, outputs):
        return -self.cv * torch.sum(targets * outputs, axis=1)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        # return torch.sum((targets - outputs) ** 2) / targets.size()[0]
        return -self.cv * torch.sum(targets * outputs) / targets.size()[0]

    @tracy_zone
    def save_checkpoint(
        self, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt",
        half_storage=True,
    ):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)

        # Convert NNArgs namedtuple to dict for safe serialization
        args_dict = self.args._asdict()

        # Convert model weights to half-precision for smaller checkpoints
        if half_storage:
            dtype = get_storage_dtype()
            state_dict_save = {
                k: to_half_safe(v, dtype) for k, v in self.nnet.state_dict().items()
            }
        else:
            state_dict_save = self.nnet.state_dict()

        buffer = io.BytesIO()
        torch.save(
            {
                "state_dict": state_dict_save,
                "opt_state": self.optimizer.state_dict(),
                "args": args_dict,  # Save as dict, not namedtuple
                "game": self.game,
                "version": "4.0",  # zstd-compressed, half-precision weights, no scheduler
            },
            buffer,
        )
        with open(filepath, 'wb') as f:
            f.write(zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=-1).compress(buffer.getvalue()))

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

        # Register safe globals for the game class
        torch.serialization.add_safe_globals([Game])

        # Try zstd decompression first, fall back to legacy uncompressed
        device = get_device()
        with open(filepath, 'rb') as f:
            raw = f.read()
        try:
            data = zstd.ZstdDecompressor().decompress(raw)
            checkpoint = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
        except zstd.ZstdError:
            checkpoint = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)

        assert checkpoint["game"] == Game, (
            f"Mismatching game type when loading model: got: {checkpoint['game'].__name__} want: {Game.__name__}"
        )

        # Reconstruct NNArgs from dict, dropping removed fields
        args_dict = checkpoint["args"]
        args_dict.pop("lr_milestone", None)
        args = NNArgs(**args_dict)

        net = NNWrapper(checkpoint["game"], args)
        # Convert half-precision weights back to float32 for training
        state_dict = {
            k: v.float() if v.is_floating_point() else v
            for k, v in checkpoint["state_dict"].items()
        }
        net.nnet.load_state_dict(state_dict)
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
