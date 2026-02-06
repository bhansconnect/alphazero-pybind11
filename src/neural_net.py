from collections import namedtuple
import os
import torch
from torch import optim, nn
from torch.autograd import profiler
from tqdm import tqdm
import numpy as np
import alphazero
from tracy_utils import tracy_zone

# This is an autotuner for network speed.
torch.backends.cudnn.benchmark = True

NNArgs = namedtuple(
    "NNArgs",
    ["num_channels", "depth", "kernel_size", "lr_milestone", "dense_net", "lr", "cv",
     "star_gambit_spatial", "use_fixup", "multi_size"],
    defaults=(40, False, 0.01, 1.5, False, False, False),
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


class FixupResidualBlock(nn.Module):
    """Fixup residual block - replaces BatchNorm with learnable biases and scaling.

    Based on "Fixup Initialization: Residual Learning Without Normalization"
    (Zhang et al., 2019). This is also used by KataGo for handling variable-size
    boards with masked inputs, where BatchNorm statistics can be corrupted.
    """

    def __init__(self, channels, depth, kernel_size=3):
        super(FixupResidualBlock, self).__init__()
        # Pre-activation biases
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        # Create conv without bias (we add our own)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding='same', bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                               padding='same', bias=False)
        # Scale factor for the residual branch
        self.scale = nn.Parameter(torch.ones(1))

        # Fixup initialization: scale down initial weights
        # For m=2 branches (conv1 and conv2), use depth^(-1/(2*m)) = depth^(-0.5)
        # Per Zhang et al. 2019 "Fixup Initialization" Theorem 1
        nn.init.normal_(self.conv1.weight, mean=0, std=np.sqrt(2 / (channels * kernel_size**2)) * depth**(-0.5))
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        residual = x
        out = x + self.bias1a
        out = self.relu1(out)
        out = self.conv1(out + self.bias1b)
        out = out + self.bias2a
        out = self.relu1(out)  # Note: reusing relu1 is fine
        out = self.conv2(out + self.bias2b)
        out = out * self.scale
        return out + residual


class FixupDenseBlock(nn.Module):
    """Fixup-style dense block without BatchNorm."""

    def __init__(self, in_channels, growth_rate, depth, bn_size=4, kernel_size=3):
        super(FixupDenseBlock, self).__init__()
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * bn_size, kernel_size=1,
                               padding='same', bias=False)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate * bn_size, growth_rate, kernel_size=kernel_size,
                               padding='same', bias=False)
        # Scale factor for output branch (learnable, initialized to 1)
        self.scale = nn.Parameter(torch.ones(1))

        # Fixup initialization
        # For m=2 branches (conv1 and conv2), use depth^(-1/(2*m)) = depth^(-0.5)
        # Per Zhang et al. 2019 "Fixup Initialization" Theorem 1
        nn.init.normal_(self.conv1.weight, mean=0, std=np.sqrt(2 / in_channels) * depth**(-0.5))
        nn.init.normal_(self.conv2.weight, mean=0, std=np.sqrt(2 / (growth_rate * bn_size * kernel_size**2)) * depth**(-0.5))

    def forward(self, x):
        out = x + self.bias1
        out = self.relu1(out)
        out = self.conv1(out)
        out = out + self.bias2
        out = self.relu2(out)
        out = self.conv2(out)
        out = out * self.scale  # Apply learnable scale
        out = torch.cat([x, out], 1)
        return out


class NNArch(nn.Module):
    def __init__(self, game, args):
        super(NNArch, self).__init__()
        # game params
        in_channels, in_x, in_y = game.CANONICAL_SHAPE()
        self.dense_net = args.dense_net
        self.star_gambit_spatial = args.star_gambit_spatial
        self.use_fixup = getattr(args, 'use_fixup', False)
        self.multi_size = getattr(args, 'multi_size', False)

        if not self.dense_net:
            self.conv1 = conv(
                in_channels, args.num_channels, kernel_size=args.kernel_size
            )
            if self.use_fixup:
                # Fixup: use bias instead of BatchNorm for first conv
                self.bn1 = None
                self.bias1 = nn.Parameter(torch.zeros(1))
            else:
                self.bn1 = nn.BatchNorm2d(args.num_channels)
                self.bias1 = None

        self.layers = []
        for i in range(args.depth):
            if self.dense_net:
                if self.use_fixup:
                    self.layers.append(
                        FixupDenseBlock(
                            in_channels + args.num_channels * i,
                            args.num_channels,
                            depth=args.depth,
                            kernel_size=args.kernel_size,
                        )
                    )
                else:
                    self.layers.append(
                        DenseBlock(
                            in_channels + args.num_channels * i,
                            args.num_channels,
                            kernel_size=args.kernel_size,
                        )
                    )
            else:
                if self.use_fixup:
                    self.layers.append(
                        FixupResidualBlock(
                            args.num_channels,
                            depth=args.depth,
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

        # Value head - with optional global pooling for multi-size support
        if self.use_fixup:
            self.v_bn = None
            self.v_bias = nn.Parameter(torch.zeros(1))
        else:
            self.v_bn = nn.BatchNorm2d(32)
            self.v_bias = None
        self.v_relu = nn.ReLU(inplace=True)

        if self.multi_size:
            # Multi-size value head uses global average pooling (works on any board size)
            self.v_global_pool = nn.AdaptiveAvgPool2d(1)  # Output: (B, 32, 1, 1)
            self.v_fc1 = nn.Linear(32, 256)
        else:
            self.v_flatten = nn.Flatten()
            self.v_fc1 = nn.Linear(32 * in_x * in_y, 256)

        self.v_fc1_relu = nn.ReLU(inplace=True)
        self.v_fc2 = nn.Linear(256, game.NUM_PLAYERS() + 1)
        self.v_softmax = nn.LogSoftmax(1)

        # Policy head with optional Fixup
        if self.use_fixup:
            self.pi_bn = None
            self.pi_bias = nn.Parameter(torch.zeros(1))
        else:
            self.pi_bn = nn.BatchNorm2d(32)
            self.pi_bias = None
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
            if self.use_fixup:
                self.pi_bn2 = None
                self.pi_bias2 = nn.Parameter(torch.zeros(1))
            else:
                self.pi_bn2 = nn.BatchNorm2d(10)
                self.pi_bias2 = None

            # Global policy: for deploy and end_turn actions
            # For multi-size, use global pooling before the global policy head
            self.pi_flatten = nn.Flatten()
            if self.multi_size:
                self.pi_global_pool = nn.AdaptiveAvgPool2d(1)
                self.pi_global = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, self.sg_global_actions),
                    nn.LayerNorm(self.sg_global_actions)
                )
            else:
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

    @tracy_zone
    def forward(self, s):
        # s: batch_size x num_channels x board_x x board_y
        # Save original input for extracting valid hex mask (channel 0) in multi_size mode
        s_original = s

        with profiler.record_function("conv-layers"):
            if not self.dense_net:
                s = self.conv1(s)
                if self.use_fixup:
                    s = s + self.bias1
                else:
                    s = self.bn1(s)
            s = self.conv_layers(s)

        with profiler.record_function("v-head"):
            v = self.v_conv(s)
            if self.use_fixup:
                v = v + self.v_bias
            else:
                v = self.v_bn(v)
            v = self.v_relu(v)

            if self.multi_size:
                # Global pooling for variable board sizes
                v = self.v_global_pool(v)  # (B, 32, 1, 1)
                v = v.view(v.size(0), -1)  # (B, 32)
            else:
                v = self.v_flatten(v)

            v = self.v_fc1(v)
            v = self.v_fc1_relu(v)
            v = self.v_fc2(v)
            v = self.v_softmax(v)

        with profiler.record_function("pi-head"):
            pi = self.pi_conv(s)
            if self.use_fixup:
                pi = pi + self.pi_bias
            else:
                pi = self.pi_bn(pi)
            pi = self.pi_relu(pi)

            if self.star_gambit_spatial:
                # Spatial policy head for Star Gambit
                # For global actions, either use flattened conv or global pooling
                if self.multi_size:
                    s_global = self.pi_global_pool(pi)  # (B, 32, 1, 1)
                    s_global = s_global.view(s_global.size(0), -1)  # (B, 32)
                else:
                    s_global = self.pi_flatten(pi)

                # Spatial actions: (B, 32, H, W) -> (B, 10, H, W) -> (B, H, W, 10) -> (B, H*W*10)
                pi_spatial = self.pi_conv2(pi)
                if self.use_fixup:
                    pi_spatial = pi_spatial + self.pi_bias2
                else:
                    pi_spatial = self.pi_bn2(pi_spatial)
                pi_spatial = pi_spatial.permute(0, 2, 3, 1)  # (B, H, W, 10)
                batch_size = pi_spatial.shape[0]

                # Apply policy masking for multi_size mode (padding positions get -inf)
                if self.multi_size:
                    # Channel 0 of input is valid hex mask (1.0 for valid, 0.0 for padding)
                    valid_hex_mask = s_original[:, 0, :, :]  # (B, H, W)
                    # Expand to (B, H, W, 10) for all action types per position
                    spatial_mask = valid_hex_mask.unsqueeze(-1).expand(-1, -1, -1, 10)
                    # Mask invalid positions with -inf (becomes 0 after log_softmax)
                    pi_spatial = pi_spatial.masked_fill(spatial_mask < 0.5, float('-inf'))

                pi_spatial = pi_spatial.reshape(batch_size, -1)  # (B, H*W*10)

                # Global actions: deploy + end_turn
                pi_global = self.pi_global(s_global)  # (B, 19)

                # Concatenate: spatial + global
                pi = torch.cat([pi_spatial, pi_global], dim=1)  # (B, H*W*10 + 19)
                pi = self.pi_softmax(pi)
            else:
                pi = self.pi_flatten(pi)
                pi = self.pi_fc1(pi)
                pi = self.pi_softmax(pi)

        return v, pi


class NNWrapper:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet = NNArch(game, args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
        )

        def lr_lambda(epoch):
            if epoch < 5:
                return 1 / 3
            elif epoch > args.lr_milestone:
                return 1 / 10
            else:
                return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=args.lr_milestones, gamma=0.1)
        self.device = get_device()
        self.cv = args.cv
        self.nnet.to(self.device)

    @tracy_zone
    def losses(self, dataset):
        self.nnet.eval()
        l_v = 0
        l_pi = 0
        for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
            canonical, target_vs, target_pis = batch
            canonical = canonical.contiguous().to(self.device, non_blocking=True)
            target_vs = target_vs.contiguous().to(self.device, non_blocking=True)
            target_pis = target_pis.contiguous().to(self.device, non_blocking=True)

            out_v, out_pi = self.nnet(canonical)
            l_v += self.loss_v(target_vs, out_v).item()
            l_pi += self.loss_pi(target_pis, out_pi).item()
        return l_v / len(dataset), l_pi / len(dataset)

    @tracy_zone
    def sample_loss(self, dataset, size):
        loss = np.zeros(size)
        self.nnet.eval()
        i = 0
        for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
            canonical, target_vs, target_pis = batch
            canonical = canonical.contiguous().to(self.device, non_blocking=True)
            target_vs = target_vs.contiguous().to(self.device, non_blocking=True)
            target_pis = target_pis.contiguous().to(self.device, non_blocking=True)

            out_v, out_pi = self.nnet(canonical)
            l_v = self.sample_loss_v(target_vs, out_v)
            l_pi = self.sample_loss_pi(target_pis, out_pi)
            total_loss = l_pi + l_v
            for sample_loss in total_loss:
                loss[i] = sample_loss.detach()
                i += 1
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

        self.scheduler.step()
        pbar.close()
        return v_loss / steps_to_train, pi_loss / steps_to_train

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    @tracy_zone
    def process(self, batch):
        batch = batch.contiguous().to(self.device, non_blocking=True)
        self.nnet.eval()
        with torch.no_grad():
            v, pi = self.nnet(batch)
            res = (torch.exp(v), torch.exp(pi))
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
        self, folder=os.path.join("data", "checkpoint"), filename="checkpoint.pt"
    ):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)

        # Convert NNArgs namedtuple to dict for safe serialization
        args_dict = self.args._asdict()

        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "sch_state": self.scheduler.state_dict(),
                "args": args_dict,  # Save as dict, not namedtuple
                "game": self.game,
                "version": "2.0",  # Add version for compatibility
            },
            filepath,
        )

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

        # Load with map_location for device flexibility
        device = get_device()
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)

        assert checkpoint["game"] == Game, (
            f"Mismatching game type when loading model: got: {checkpoint['game'].__name__} want: {Game.__name__}"
        )

        # Reconstruct NNArgs from dict
        args = NNArgs(**checkpoint["args"])

        net = NNWrapper(checkpoint["game"], args)
        net.nnet.load_state_dict(checkpoint["state_dict"])
        net.optimizer.load_state_dict(checkpoint["opt_state"])
        net.scheduler.load_state_dict(checkpoint["sch_state"])
        return net


def bench_network():

    Game = alphazero.OpenTaflGS
    depth = 4
    channels = 12
    dense_net = True
    batch_size = 1024
    nnargs = NNArgs(num_channels=channels, depth=depth, kernel_size=3, dense_net=dense_net)

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
