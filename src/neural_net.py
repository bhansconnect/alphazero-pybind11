from collections import namedtuple
import os
import torch
from torch import optim, nn
from torch.autograd import profiler
from tqdm import tqdm

NNArgs = namedtuple('NNArgs', ['num_channels', 'depth', 'lr_milestones',
                               'lr', 'cuda'], defaults=([40], 0.001, torch.cuda.is_available()))


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

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

        self.conv1 = conv3x3(in_channels, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(ResidualBlock(
                args.num_channels, args.num_channels))
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv1x1(args.num_channels, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_relu = nn.ReLU(inplace=True)
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(in_x*in_y,
                               in_x*in_y//2)
        self.v_fc1_relu = nn.ReLU(inplace=True)
        self.v_fc2 = nn.Linear(in_x*in_y//2, game.NUM_PLAYERS())
        self.v_tanh = nn.Tanh()

        self.pi_conv = conv1x1(args.num_channels, 2)
        self.pi_bn = nn.BatchNorm2d(2)
        self.pi_relu = nn.ReLU(inplace=True)
        self.pi_flatten = nn.Flatten()
        self.pi_fc1 = nn.Linear(in_x*in_y*2, game.NUM_MOVES())
        self.pi_softmax = nn.LogSoftmax(1)

    def forward(self, s):
        # s: batch_size x num_channels x board_x x board_y
        with profiler.record_function("resnet"):
            s = self.bn1(self.conv1(s))
            s = self.resnet(s)

        with profiler.record_function("v-head"):
            v = self.v_conv(s)
            v = self.v_bn(v)
            v = self.v_relu(v)
            v = self.v_flatten(v)
            v = self.v_fc1(v)
            v = self.v_fc1_relu(v)
            v = self.v_fc2(v)
            v = self.v_tanh(v)

        with profiler.record_function("pi-head"):
            pi = self.pi_conv(s)
            pi = self.pi_bn(pi)
            pi = self.pi_relu(pi)
            pi = self.pi_flatten(pi)
            pi = self.pi_fc1(pi)
            pi = self.pi_softmax(pi)

        return v, pi


class NNWrapper:
    def __init__(self, game, args):
        self.nnet = NNArch(game, args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_milestones, gamma=0.1)
        self.cuda = args.cuda
        if self.cuda:
            self.nnet.cuda()

    def train(self, batches, train_steps):
        self.nnet.train()

        v_loss = 0
        pi_loss = 0
        current_step = 0
        pbar = tqdm(total=train_steps, unit='batches', desc='Training NN')
        while current_step < train_steps:
            for batch in batches:
                if current_step == train_steps:
                    break
                canonical, target_vs, target_pis = batch
                if self.cuda:
                    canonical = canonical.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v, out_pi = self.nnet(canonical)
                l_v = self.loss_v(target_vs, out_v)
                l_pi = self.loss_pi(target_pis, out_pi)
                total_loss = l_pi + l_v
                total_loss.backward()
                self.optimizer.step()

                # record loss and update progress bar.
                pi_loss += l_pi.item()
                v_loss += l_v.item()
                current_step += 1
                pbar.set_postfix(
                    {'v loss': v_loss/current_step, 'pi loss': pi_loss/current_step})
                pbar.update()

        self.scheduler.step()
        pbar.close()
        return v_loss/train_steps, pi_loss/train_steps

    def predict(self, canonical):
        v, pi = self.process(canonical.unsqueeze(0))
        return v[0], pi[0]

    def process(self, batch):
        if self.cuda:
            batch = batch.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            v, pi = self.nnet(batch)
            return v, torch.exp(pi)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='data/checkpoint', filename='checkpoint.pt'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict()
        }, filepath)

    def load_checkpoint(self, folder='data/checkpoint', filename='checkpoint.pt'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_state'])
        self.scheduler.load_state_dict(checkpoint['sch_state'])
