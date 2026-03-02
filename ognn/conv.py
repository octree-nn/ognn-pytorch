import torch
import math
import torch.nn.functional as F

from ognn.octreed import OctreeD, Graph
from ognn.utils import scatter_mean


class GraphConv(torch.nn.Module):

  def __init__(
          self, in_channels: int, out_channels: int, n_edge_type: int = 7,
          n_node_type: int = 0, use_bias: bool = False):
    super().__init__()
    self.avg_degree = 7
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_edge_type = n_edge_type
    self.n_node_type = n_node_type
    self.use_bias = use_bias

    node_channel = n_node_type if n_node_type > 1 else 0
    self.weights = torch.nn.Parameter(
        torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels))
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
        x[col], index, dim=0, dim_size=x.shape[0] * self.n_edge_type)

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
    return ('in_channels={}, out_channels={}, n_edge_type={}, n_node_type={}, '
            'use_bias={}'.format(self.in_channels, self.out_channels,
             self.n_edge_type, self.n_node_type, self.use_bias))  # noqa


def im2col_simplified(graph: Graph, n_edge_type: int, data: torch.Tensor):
    N, F_dim = data.shape
    device = data.device
    
    # 1. 使用 F.pad 替代 cat+zeros
    # F.pad 比 torch.cat 更快，因为它不需要显式创建一个 (1, F) 的全0张量对象
    # 参数 (0, 0, 0, 1) 表示：最后一维左右各垫0个，倒数第二维(行)上边垫0个、下边垫1个
    data_padded = F.pad(data, (0, 0, 0, 1))
    
    # 2. 预分配目标索引
    # 全指向最后一个位置 (即 data_padded 的最后一行，全是 0)
    target_idx = torch.full((N * n_edge_type,), N, dtype=torch.long, device=device)
    
    # 3. 准备索引数据
    row, col = graph.edge_idx
    # 计算一维扁平索引
    index = torch.add(graph.edge_type, row, alpha=n_edge_type)
    
    # 【优化】使用 scatter_ 替代 target_idx[index] = col
    # 避免了左值索引的 Python overhead，直接调用底层核函数进行赋值
    target_idx.scatter_(0, index, col)
    target_idx[n_edge_type - 1 :: n_edge_type] = torch.arange(N, device=device)
    col_data = torch.index_select(data_padded, 0, target_idx)
    
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

        # add bias
        if self.use_bias:
            output += self.bias
        return output