# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Chuan-Zhi Zhou, Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import triton
import ocnn
from ocnn.octree import Points, Octree
from ognn.conv import GraphConv, GraphConvNew, GraphConvIGEMM
from ognn.octreed import OctreeD, simplify_graph_forward_inplace

device = "cuda"
depth2channel = {4: 512, 5: 256, 6: 128, 7: 128, 8: 64, 9: 32, 10: 32}


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


# Number-of-voxels at different depths:
#      depth      Number-of-voxels
# 0      5.0            3840
# 1      6.0           15192
# 2      7.0           64160
# 3      8.0          252392
# 4      9.0         1026536
# 5     10.0         4113056


configs = [
    triton.testing.Benchmark(
        x_names=["depth"],
        x_vals=[5, 6, 7, 8, 9, 10],
        line_arg="provider",
        line_vals=["original", 'original_simplify', "egemm", "igemm"],
        line_names=["Original", "Original Simplify", "EGEMM", "IGEMM"],
        styles=[("green", "-"), ("red", "-"), ("blue", "-"), ("purple", "-")],
        ylabel="Latency (ms)",
        plot_name=f"{mode}-{str(dtype)}",
        args={"mode": mode, "dtype": dtype},
        y_log=True,
    )
    for mode in ["fwd", "bwd"]
    for dtype in [torch.bfloat16]
    # for dtype in [torch.float32, torch.bfloat16]
]


@triton.testing.perf_report(configs)
def benchmark(depth, provider, mode, dtype):
    in_channel = depth2channel[depth]
    out_channel = in_channel

    # Create modules
    if provider == "original" or provider == "original_simplify":
        model = (
            GraphConv(
                in_channel,
                out_channel,
                use_bias=True,
            )
            .type(dtype)
            .to(device)
        )

    elif provider == "egemm":
        model = (
            GraphConvNew(
                in_channel,
                out_channel,
                use_bias=True,
            )
            .type(dtype)
            .to(device)
        )
    elif provider == "igemm":
        model = (
            GraphConvIGEMM(
                in_channel,
                out_channel,
                use_bias=True,
            )
            .type(dtype)
            .to(device)
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    # Generate coordinates and octree
    reso = 2**depth
    pos = sphere_coords(2**depth, device=device)
    pos = pos / reso * 2 - 1
    octree = Octree(depth, 2, device=device)
    octree.build_octree(Points(pos))
    octree.construct_all_neigh()
    octree = OctreeD(octree)
    nnum = octree.graphs[depth].nnum
    if provider == "original_simplify" or provider == "egemm" or provider == "igemm":
        for graph in octree.graphs:
            simplify_graph_forward_inplace(graph)

    data = torch.randn(
        nnum,
        in_channel,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    if mode == "fwd":

        def run_fwd():
            return model(data, octree, depth)

        ms = triton.testing.do_bench(run_fwd)

    else:
        def run_bwd():
            out = model(data, octree, depth)
            grad_out = torch.randn_like(out)
            out.backward(grad_out)

        ms = triton.testing.do_bench(run_bwd)
    return ms


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(__file__))
    rst_path = os.path.join(curr_path, "benchmark")
    os.makedirs(rst_path, exist_ok=True)
    benchmark.run(print_data=True, show_plots=False, save_path=rst_path)
