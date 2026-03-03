import torch
import math
import torch.nn.functional as F

from ognn.octreed import OctreeD, Graph
from ognn.utils import scatter_mean
from ognn.kernels import conv_fwd_implicit_gemm_splitk, conv_bwd_implicit_gemm_splitk


class FlexGEMMFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        neigh: torch.Tensor,
    ):
        data = data.contiguous()
        weights = weights.contiguous()
        weights = weights.to(data.dtype)  # for torch.amp
        neigh = neigh.contiguous()
        if bias is not None:
            bias = bias.contiguous()
            bias = bias.to(data.dtype)  # for torch.amp

        out = conv_fwd_implicit_gemm_splitk(data, weights, bias, neigh)
        ctx.save_for_backward(data, weights, bias, neigh)
        return out

    @staticmethod
    def backward(ctx, grad):
        data, weights, bias, neigh = ctx.saved_tensors
        grad = grad.contiguous()
        grad_input, grad_weight, grad_bias = conv_bwd_implicit_gemm_splitk(
            grad, data, weights, bias, neigh, ctx.needs_input_grad
        )
        return grad_input, grad_weight, grad_bias, None


flex_gemm_fn = FlexGEMMFn.apply


class GraphConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_edge_type: int = 7,
        n_node_type: int = 0,
        use_bias: bool = False,
    ):
        super().__init__()
        self.avg_degree = 7
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_edge_type = n_edge_type
        self.n_node_type = n_node_type
        self.use_bias = use_bias

        node_channel = n_node_type if n_node_type > 1 else 0
        self.weights = torch.nn.Parameter(
            torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels)
        )
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.avg_degree * self.in_channels
        fan_out = self.avg_degree * self.out_channels
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(self.weights, -a, a)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
        graph = octree.graphs[depth]

        # concatenate the one_hot vector for node_type
        if self.n_node_type > 1:
            one_hot = F.one_hot(graph.node_type, num_classes=self.n_node_type)
            x = torch.cat([x, one_hot], dim=1)

        # x -> col_data
        row, col = graph.edge_idx
        index = row * self.n_edge_type + graph.edge_type
        col_data = scatter_mean(
            x[col], index, dim=0, dim_size=x.shape[0] * self.n_edge_type
        )

        # add self-loops
        index = torch.arange(graph.nnum, dtype=torch.int64, device=x.device)
        index = index * self.n_edge_type + (self.n_edge_type - 1)
        col_data[index] = x

        # matrix product
        output = col_data.view(x.shape[0], -1) @ self.weights

        # add bias
        if self.use_bias:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return (
            "in_channels={}, out_channels={}, n_edge_type={}, n_node_type={}, "
            "use_bias={}".format(
                self.in_channels,
                self.out_channels,
                self.n_edge_type,
                self.n_node_type,
                self.use_bias,
            )
        )  # noqa


@torch.compile(dynamic=True)
def im2col_simplified(graph: Graph, n_edge_type: int, data: torch.Tensor):
    N, F_dim = data.shape
    device = data.device

    data_padded = F.pad(data, (0, 0, 0, 1))

    target_idx = torch.full((N * n_edge_type,), N, dtype=torch.long, device=device)

    row, col = graph.edge_idx
    index = torch.add(graph.edge_type, row, alpha=n_edge_type)
    target_idx.scatter_(0, index, col)
    target_idx[n_edge_type - 1 :: n_edge_type] = torch.arange(N, device=device)
    col_data = F.embedding(target_idx, data_padded, padding_idx=N)

    return col_data.view(N, -1)


class GraphConvNew(GraphConv):
    def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
        graph = octree.graphs[depth]

        # concatenate the one_hot vector for node_type
        if self.n_node_type > 1:
            one_hot = F.one_hot(graph.node_type, num_classes=self.n_node_type)
            x = torch.cat([x, one_hot], dim=1)

        # x -> col_data
        col_data = im2col_simplified(graph, self.n_edge_type, x)

        # matrix product
        output = col_data @ self.weights

        # # add bias
        if self.use_bias:
            output += self.bias
        return output


class GraphConvIGEMM(GraphConv):
    def forward(self, x: torch.Tensor, octree: OctreeD, depth: int):
        graph = octree.graphs[depth]

        # concatenate the one_hot vector for node_type
        if self.n_node_type > 1:
            one_hot = F.one_hot(graph.node_type, num_classes=self.n_node_type)
            x = torch.cat([x, one_hot], dim=1)
        
        if hasattr(graph, 'neighbour'):
            neigh = graph.neighbour
        else:
            N = graph.nnum
            target_idx = torch.full((N * self.n_edge_type,), -1, dtype=torch.long, device=x.device)
            row, col = graph.edge_idx
            index = torch.add(graph.edge_type, row, alpha=self.n_edge_type)
            target_idx.scatter_(0, index, col)
            target_idx[self.n_edge_type - 1 :: self.n_edge_type] = torch.arange(N, device=x.device)
            neigh = target_idx.reshape(N, -1).contiguous()
        
        output = flex_gemm_fn(x, self.weights.reshape(self.n_edge_type, -1, self.out_channels).permute(2, 0, 1), self.bias, neigh)
        return output
