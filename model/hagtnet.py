"""
HAGT-Net: Hierarchical Adaptive Graph Topology Network.

A skeleton-based emotion recognition architecture combining:
  - a center-of-mass hierarchical graph structure, organizing joints into
    concentric rings and propagating information both outward and inward
    along the hierarchy;
  - an adaptive, sample-dependent topology refinement mechanism, which
    adjusts joint-to-joint connection strength per input rather than using a
    single fixed connectivity pattern;
  - a local dynamic feature-space branch that builds its own nearest-neighbor
    graph directly from the current feature map;
  - a hierarchy-level attention module that reweights the contribution of
    each ring-pair before the layers are combined;
  - a multimodal fusion stage that injects an auxiliary affective feature
    vector into the network at a fixed depth via elementwise gating.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from graph.tools import get_groups


def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


# ---------------------------------------------------------------------------
# Temporal modeling
# ---------------------------------------------------------------------------

class TemporalConv(nn.Module):
    """Single dilated 1D temporal convolution branch."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) + self.bias
        x = self.bn(x)
        return x


class MultiScaleTemporalConv(nn.Module):
    """Multi-branch temporal convolution: several dilated convs, a pooling
    branch, and a 1x1 shortcut branch, concatenated along the channel axis."""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 dilations=(1, 2), residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "out_channels must be a multiple of num_branches"

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class ResidualConv(nn.Module):
    """Plain temporal convolution used for residual/skip projections."""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualConv, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# Local dynamic feature-space branch
# ---------------------------------------------------------------------------

class LocalGraphConv(nn.Module):
    """
    Builds a nearest-neighbor graph directly in feature space (per sample)
    and convolves over each joint's local neighborhood. Independent of the
    fixed hierarchical structure — this branch discovers its own local
    connectivity from the current feature map.
    """

    def __init__(self, in_channels, out_channels, k):
        super(LocalGraphConv, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, dim=4):
        if dim == 3:
            N, C, L = x.size()
        else:
            N, C, T, V = x.size()
            x = x.mean(dim=-2, keepdim=False)

        x = self._get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]

        if dim != 3:
            x = repeat(x, "n c v -> n c t v", t=T)
        return x

    def _knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def _get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self._knn(x, k=k)
        device = x.device

        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * V
        idx = idx + idx_base
        idx = idx.view(-1)

        x = rearrange(x, "n c v -> n v c")
        feature = rearrange(x, "n v c -> (n v) c")[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, "n v c -> n v k c", k=k)

        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, "n v k c -> n c v k")
        return feature


# ---------------------------------------------------------------------------
# Hierarchy-level attention
# ---------------------------------------------------------------------------

class HierarchyAttention(nn.Module):
    """
    Reweights each hierarchy layer's contribution before they are combined,
    using a lightweight local-graph attention computed over per-layer,
    ring-pooled features.
    """

    def __init__(self, in_channels, num_layers, CoM):
        super(HierarchyAttention, self).__init__()
        self.num_layers = num_layers

        groups = get_groups(CoM=CoM)
        for i, group in enumerate(groups):
            groups[i] = [j - 1 for j in group]

        inter_channels = in_channels // 4
        self.layers = [groups[i] + groups[i + 1] for i in range(len(groups) - 1)]

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.local_graph_conv = LocalGraphConv(inter_channels, inter_channels, k=3)
        self.aggregate = nn.Conv1d(inter_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, L, T, V = x.size()

        x_t = x.max(dim=-2, keepdim=False)[0]
        x_t = self.conv_down(x_t)

        x_sampled = []
        for i in range(self.num_layers):
            s_t = x_t[:, :, i, self.layers[i]]
            s_t = s_t.mean(dim=-1, keepdim=True)
            x_sampled.append(s_t)
        x_sampled = torch.cat(x_sampled, dim=2)

        att = self.local_graph_conv(x_sampled, dim=3)
        att = self.aggregate(att).view(N, C, L, 1, 1)

        out = (x * self.sigmoid(att)).sum(dim=2, keepdim=False)
        return out


# ---------------------------------------------------------------------------
# Adaptive topology refinement
# ---------------------------------------------------------------------------

class AdaptiveTopologyConv(nn.Module):
    """
    Sample-adaptive graph convolution. Rather than propagating features
    along a single fixed adjacency, this module computes a per-sample,
    per-channel-group correlation between joints and uses it to refine a
    static base topology before propagation, so the effective connectivity
    can flex per input on top of the fixed structural prior.
    """

    def __init__(self, in_channels, out_channels, rel_reduction=4, mid_reduction=1):
        super(AdaptiveTopologyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels in (3, 9):
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, base_topology=None, alpha=1):
        summary_a = self.conv1(x).mean(-2)
        summary_b = self.conv2(x).mean(-2)
        content = self.conv3(x)

        relation = self.tanh(summary_a.unsqueeze(-1) - summary_b.unsqueeze(-2))
        refined_topology = self.conv4(relation) * alpha
        if base_topology is not None:
            refined_topology = refined_topology + base_topology.unsqueeze(0).unsqueeze(0)

        out = torch.einsum("ncuv,nctv->nctu", refined_topology, content)
        return out


# ---------------------------------------------------------------------------
# Hierarchical graph convolution block
# ---------------------------------------------------------------------------

class HierarchicalGraphConv(nn.Module):
    """
    Spatial graph convolution operating over the multi-layer hierarchical
    topology. For each hierarchy layer, three directional sub-relations
    (identity / forward / reverse) are each processed by an
    AdaptiveTopologyConv module, run in parallel with a LocalGraphConv
    branch operating directly on the current feature map. All branch
    outputs are concatenated, and HierarchyAttention combines the
    per-layer results into the block's final output.
    """

    def __init__(self, in_channels, out_channels, A, residual=True, use_attention=False,
                 CoM=0, topology_rel_reduction=4):
        super(HierarchicalGraphConv, self).__init__()
        self.num_layers = A.shape[0]
        self.num_subset = A.shape[1]
        self.use_attention = use_attention

        inter_channels = out_channels // (self.num_subset + 1)

        self.base_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_down = nn.ModuleList()
        self.branches = nn.ModuleList()
        for i in range(self.num_layers):
            branch = nn.ModuleList()
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            ))
            for j in range(self.num_subset):
                branch.append(AdaptiveTopologyConv(inter_channels, inter_channels, rel_reduction=topology_rel_reduction))
            branch.append(LocalGraphConv(inter_channels, inter_channels, k=5))
            self.branches.append(branch)

        if self.use_attention:
            self.hierarchy_attention = HierarchyAttention(out_channels, num_layers=self.num_layers, CoM=CoM)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        A = self.base_topology
        out = []
        for i in range(self.num_layers):
            layer_out = []
            x_down = self.conv_down[i](x)
            for j in range(self.num_subset):
                z = self.branches[i][j](x_down, A[i, j], self.alpha)
                layer_out.append(z)
            layer_out.append(self.branches[i][-1](x_down))
            out.append(torch.cat(layer_out, dim=1))

        out = torch.stack(out, dim=2)
        if self.use_attention:
            out = self.hierarchy_attention(out)
        else:
            out = out.sum(dim=2, keepdim=False)

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)
        return out


class SpatioTemporalBlock(nn.Module):
    """One full block: hierarchical spatial graph convolution followed by
    multi-scale temporal convolution, with a residual connection."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 kernel_size=5, dilations=(1, 2), use_attention=True, CoM=0,
                 topology_rel_reduction=4):
        super(SpatioTemporalBlock, self).__init__()
        self.spatial = HierarchicalGraphConv(
            in_channels, out_channels, A, use_attention=use_attention, CoM=CoM,
            topology_rel_reduction=topology_rel_reduction,
        )
        self.temporal = MultiScaleTemporalConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilations=dilations, residual=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = ResidualConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.temporal(self.spatial(x)) + self.residual(x))


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class HAGTNet(nn.Module):
    """
    Hierarchical Adaptive Graph Topology Network.

    Stacks 10 SpatioTemporalBlocks over a channel progression of
    64-64-64-64-128-128-128-256-256-256, with temporal downsampling at
    blocks 5 and 8. An auxiliary affective feature vector is optionally
    fused into the network via elementwise gating at a configurable depth
    (default: after block 7).
    """

    def __init__(self, num_class=4, num_point=16, num_person=1, graph=None,
                 graph_args=None, in_channels=3, drop_out=0, adaptive=True,
                 aff_dim=None, topology_rel_reduction=4):
        super(HAGTNet, self).__init__()

        if graph is None:
            raise ValueError("graph must be specified")
        Graph = import_class(graph)
        self.graph = Graph(**(graph_args or {}))
        A, CoM = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.aff_dim = aff_dim
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channels = 64
        mid_channels = base_channels * 2
        final_channels = base_channels * 4

        self.block1 = SpatioTemporalBlock(in_channels, base_channels, A, residual=False, use_attention=False, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block2 = SpatioTemporalBlock(base_channels, base_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block3 = SpatioTemporalBlock(base_channels, base_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block4 = SpatioTemporalBlock(base_channels, base_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block5 = SpatioTemporalBlock(base_channels, mid_channels, A, stride=2, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block6 = SpatioTemporalBlock(mid_channels, mid_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block7 = SpatioTemporalBlock(mid_channels, mid_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block8 = SpatioTemporalBlock(mid_channels, final_channels, A, stride=2, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block9 = SpatioTemporalBlock(final_channels, final_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)
        self.block10 = SpatioTemporalBlock(final_channels, final_channels, A, CoM=CoM, topology_rel_reduction=topology_rel_reduction)

        if self.aff_dim is not None:
            self.affect_proj = nn.Sequential(
                nn.Linear(self.aff_dim, mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, mid_channels),
            )

        self.classifier = nn.Linear(final_channels, num_class)

        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else (lambda x: x)

    def forward(self, x, aff=None):
        N, C, T, V, M = x.size()
        x = rearrange(x, "n c t v m -> n (m v c) t")
        x = self.data_bn(x)
        x = rearrange(x, "n (m v c) t -> (n m) c t v", m=M, v=V)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        if self.aff_dim is not None and aff is not None:
            gate = self.affect_proj(aff).unsqueeze(-1).unsqueeze(-1)
            x = x * gate

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.classifier(x)
