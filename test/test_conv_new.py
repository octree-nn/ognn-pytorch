# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang, Chuan-Zhi Zhou
# --------------------------------------------------------

import os
import torch
import unittest

import ocnn
from ocnn.octree import Points, Octree
from ognn.conv import GraphConv, GraphConvNew
from ognn.octreed import OctreeD, simplify_graph_forward_inplace


def sphere_coords(resolution, device="cuda"):
    r"""This function generates random features and integer coordinates for
    voxels on a thin spherical shell inside a cubic grid of resolution
    `res`. It iterates in n^3 chunks to keep memory bounded, building 3D
    meshes via `torch.meshgrid` and shifting them into global coordinates.

    Args:
      resolution: int
        The resolution of the cubic grid.
      device: str
        The device where the tensors are allocated.
    """

    n = 128
    out = []
    for i in range(0, resolution, n):
        for j in range(0, resolution, n):
            for k in range(0, resolution, n):
                block = torch.stack(
                    torch.meshgrid(
                        torch.arange(i, min(i + n, resolution), device=device),
                        torch.arange(j, min(j + n, resolution), device=device),
                        torch.arange(k, min(k + n, resolution), device=device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).int()
                dist = ((block.float() - resolution / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
                active = (dist <= resolution / 2) & (dist >= resolution / 2 - 1.25)
                out.append(block[active])
    pos = torch.cat(out, dim=0)
    return pos


class TestOctreeConvTriton(unittest.TestCase):
    def test_conv(self):
        octree = self.build_octree()
        depth2channel = {3: 1024, 4: 512, 5: 256, 6: 128, 7: 64}
        for d in [octree.depth, octree.depth - 1]:
            for out_ratio in [1.0, 0.5, 2.0]:
                self.conv_forward_backward(d, out_ratio, octree, depth2channel[d])

    def test_conv_small_channel(self):
        octree = self.build_octree()
        for d in [octree.depth]:
            for out_ratio in [1.0, 2.0]:
                for channel in [4, 8, 16]:
                    # print(f'Testing depth={d}, out_ratio={out_ratio}, channel={channel}')
                    self.conv_forward_backward(d, out_ratio, octree, channel)

    def test_conv_irregular_channel(self):
        octree = self.build_octree()
        for d in [octree.depth]:
            for out_ratio in [1.0, 2.0]:
                for channel in [2**i + 5 for i in range(6, 8)]:
                    # print(f'Testing depth={d}, out_ratio={out_ratio}, channel={channel}')
                    self.conv_forward_backward(d, out_ratio, octree, channel)

    def build_octree(self):
        r = 64
        depth, full_depth = 7, 3
        pos = sphere_coords(64, device="cuda")
        pos = pos / r * 2.0 - 1.0  # normalize to [-1,1]
        points = Points(points=pos)
        octree = Octree(depth, full_depth, device="cuda")
        octree.build_octree(points)
        octree.construct_all_neigh()
        octree = OctreeD(octree)
        for graph in octree.graphs:
            simplify_graph_forward_inplace(graph)
        return octree

    def conv_forward_backward(self, depth, out_ratio, octree, in_channel):
        atol = 5e-3
        kernel_size = [3, 3, 3]
        nempty = False
        device = "cuda"
        out_channel = int(in_channel * out_ratio)

        # initialize conv layers
        conv_original = GraphConv(
            in_channel,
            out_channel,
            use_bias=True,
        ).to(device)
        conv_new = GraphConvNew(
            in_channel,
            out_channel,
            use_bias=True,
        ).to(device)
        with torch.no_grad():
            conv_new.weights.copy_(conv_original.weights)
            conv_new.bias.copy_(conv_original.bias)

        # initialize data and grad
        data = torch.randn(octree.graphs[depth].nnum, in_channel, device=device)
        data_original = data.detach().clone().requires_grad_()
        data_new = data.detach().clone().requires_grad_()
        grad = torch.randn(octree.graphs[depth].nnum, out_channel, device=device)

        # forward
        out_original = conv_original(data_original, octree, depth)
        out_new = conv_new(data_new, octree, depth)

        # backward
        loss_original = (out_original * grad).sum()
        loss_new = (out_new * grad).sum()
        loss_original.backward()
        loss_new.backward()

        # check results
        msg = f"depth: {depth}, out_ratio: {out_ratio}"
        self.assertTrue(torch.allclose(out_original, out_new, atol=atol), msg)
        self.assertTrue(torch.allclose(data_original.grad, data_new.grad, atol=atol), msg)
        # TODO: depth: 7, out_ratio: 2.0, error: 0.0031270573381334543
        err = f", error: {self.calc_err(conv_original.weights.grad, conv_new.weights.grad)}"
        self.assertTrue(
            torch.allclose(conv_original.weights.grad, conv_new.weights.grad, atol=1e-2),
            msg + err,
        )
        self.assertTrue(
            torch.allclose(conv_original.bias.grad, conv_new.bias.grad, atol=atol), msg
        )

    def calc_err(self, src, ref):
        abs_err = (src - ref).float().abs()
        return abs_err.max().item()  # , err.mean().item()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    unittest.main()
