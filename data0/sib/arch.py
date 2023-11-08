import torch
from torch_geometric.nn import GraphConv, LayerNorm, GENConv
from torch.nn import functional as F
from torch import nn
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_sparse import SparseTensor
from typing import List, Optional, Tuple, Union
from torch import Tensor


def new_forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
    """"""
    if isinstance(x, Tensor):
        x: OptPairTensor = (x, x)

    if hasattr(self, 'lin_src'):
        x = (self.lin_src(x[0]), x[1])

    if isinstance(edge_index, SparseTensor):
        edge_attr = edge_index.storage.value().view(-1,1) 

    if edge_attr is not None and hasattr(self, 'lin_edge'):
        edge_attr = self.lin_edge(edge_attr)

    # Node and edge feature dimensionalites need to match.
    if edge_attr is not None:
        assert x[0].size(-1) == edge_attr.size(-1)

    # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    if hasattr(self, 'lin_aggr_out'):
        out = self.lin_aggr_out(out)

    if hasattr(self, 'msg_norm'):
        out = self.msg_norm(x[1] if x[1] is not None else x[0], out)

    x_dst = x[1]
    if x_dst is not None:
        if hasattr(self, 'lin_dst'):
            x_dst = self.lin_dst(x_dst)
        out = out + x_dst

    return self.mlp(out)


GENConv.forward = new_forward


class GraphConvTwoDirection(torch.nn.Module):
    def __init__(self, left_dim, right_dim, out_dim):
        super().__init__()
        # left -- constraints
        # right -- variables
        # assume out_dim is the same for left and right part
        self.left2right = GraphConv((left_dim, right_dim), out_dim, #aggr='mean', 
                                    # bias_initializer='zeros'
                                    )
        self.right2left = GraphConv((right_dim, left_dim), out_dim, #aggr='mean', 
                                    # bias_initializer='zeros'
                                    )
        # self.lin_edge=nn.Linear(1,1) 

    def forward(self, left_feas, right_feas, edge_index, edge_weight=None):
        # may try: sync or async
        # edge_index: sparse tensor (CSR) -- left to right directed edge
        # edge_index.set_value_(torch.ones(edge_index.nnz()).cuda(), layout='coo') 
        # edge_attr= edge_index.storage.value().view(-1,1) 
        # edge_attr = self.lin_edge(edge_attr) 
        l2r, r2l = edge_index, edge_index.t()
        # need an extra transpose, refer:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        l2r, r2l = r2l, l2r  # equivalent to l2r,r2l = l2r.t(),r2l.t()
        right_updated = self.left2right((left_feas, right_feas),
                                        l2r,
                                        edge_weight)  # when use sparse tensor, edge_weight is redundant. Can be deleted
        left_updated = self.right2left((right_feas, left_feas),
                                       r2l,
                                       edge_weight)
        return left_updated, right_updated


class GENConvTwoDirection(torch.nn.Module):
    def __init__(self, left_dim, right_dim, out_dim):
        super().__init__()
        kwargs = dict(aggr='softmax',
                    t=1.0, learn_t=True, num_layers=2, norm='layer', #None,  
                    edge_dim=1, 
                    )
        self.left2right = GENConv((left_dim, right_dim), out_dim, **kwargs)
        self.right2left = GENConv((right_dim, left_dim), out_dim, **kwargs)

    def forward(self, left_feas, right_feas, edge_index, ):
        l2r, r2l = edge_index, edge_index.t()
        # need an extra transpose, refer:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        l2r, r2l = r2l, l2r  # equivalent to l2r,r2l = l2r.t(),r2l.t()  
        right_updated = self.left2right((left_feas, right_feas),
                                        l2r,
                                        )  # when use sparse tensor, edge_weight is redundant. Can be deleted
        left_updated = self.right2left((right_feas, left_feas),
                                       r2l,
                                       )
        return left_updated, right_updated


class GCNBase(torch.nn.Module):
    def save(self, pn):
        torch.save(self.state_dict(), pn)

    def load(self, pn):
        st = torch.load(pn, map_location=torch.device('cpu'))
        self.load_state_dict(st)

class GCNRand(GCNBase):
    def forward(self, batch):
        x_left, x_right, edge_index, = batch.x_s, batch.x_t, batch.edge_index,
        ncons = x_left.shape[0]
        nvars = x_right.shape[0]
        dev = x_left.device
        left = torch.rand(ncons, 3).to(dev)
        right = torch.rand(nvars, 3).to(dev)
        left, right = add_knowledge(left, right, x_left, x_right)
        # left, right = F.log_softmax(left, dim=1), F.log_softmax(right, dim=1)
        return left, right

    
def add_knowledge(left_logit, right_logit, left_feas, right_feas, bound=10):
    cons_l_mask, cons_u_mask = left_feas[:, -3].abs().bool(), left_feas[:, -1].abs().bool()
    # abs because: tag for l_i=-inf is -1
    vars_l_mask, vars_u_mask = right_feas[:, -3].abs().bool(), right_feas[:, -1].abs().bool()
    # model.half -> nan in left_logit; may fix it; and maintain bound 
    left_logit = F.normalize(left_logit) * 10 
    right_logit = F.normalize(right_logit) * 10 # todo 

    left_logit[cons_l_mask, 0] = left_logit[cons_l_mask, 0] - bound 
    left_logit[cons_u_mask, 2] = left_logit[cons_u_mask, 2] - bound
    right_logit[vars_l_mask, 0] = right_logit[vars_l_mask, 0] - bound 
    right_logit[vars_u_mask, 2] = right_logit[vars_u_mask, 2] - bound 
    return left_logit, right_logit


class GCN(GCNBase):
    def __init__(self, p, q, hids, *args, **kwargs):
        super().__init__()
        self.conv1 = GraphConvTwoDirection(p, q, hids)
        self.conv2 = GraphConvTwoDirection(hids, hids, hids)
        self.conv3 = GraphConvTwoDirection(hids, hids, 3)

    def forward(self, batch):
        x_left, x_right, edge_index, = batch.x_s, batch.x_t, batch.edge_index,
        left, right = self.conv1(x_left, x_right, edge_index, )
        left, right = left.relu(), right.relu()

        left, right = self.conv2(left, right, edge_index, )
        left, right = left.relu(), right.relu()
        left = F.dropout(left, p=.1, training=self.training)
        right = F.dropout(right, p=.1, training=self.training)

        left, right = self.conv3(left, right, edge_index, )
        left, right = add_knowledge(left, right, x_left, x_right)
        # left, right = F.log_softmax(left, dim=1), F.log_softmax(right, dim=1)
        return left, right


class GCN_FC(GCNBase):
    def __init__(self, p, q, hids=128, depth=3, dp=.1, *args, **kwargs):
        super().__init__()
        self.conv1 = GraphConvTwoDirection(p, q, hids)
        self.layers = nn.ModuleList()
        for _ in range(depth-2):
            conv = GraphConvTwoDirection(hids, hids, hids)
            self.layers.append(conv)
        #----------------------------------------------------------
        self.lin_left = torch.nn.Linear(hids, 1, )
        self.lin_right = torch.nn.Linear(hids, 1, )
        self.dp=dp 

    def forward(self, batch):
        x_left, x_right, edge_index, = batch.x_s, batch.x_t, batch.edge_index,
        left, right = self.conv1(x_left, x_right, edge_index, )
        left.relu_(), right.relu_()

        for conv in self.layers:
            left, right = conv(left, right, edge_index, )
            left = F.dropout(left, p=self.dp, training=self.training)
            right = F.dropout(right, p=self.dp, training=self.training)
            left.relu_(), right.relu_()

        #left, right = self.lin_left(left), self.lin_right(right)
        left, right = self.lin_left(left).squeeze(), self.lin_right(right).squeeze()
        #-------------------------------------------------------------
        #left, right = add_knowledge(left, right, x_left, x_right)
        # left, right = F.log_softmax(left, dim=-1), F.log_softmax(right, dim=-1)
        return left, right

# class LinearTwoDirection()

# class MLP(GCNBase): 
#     def __init__(self, p, q, hids=128, depth=3, *args, **kwargs) -> None: 
#         super().__init__() 

class DeepGCNLayer(torch.nn.Module):
    r"""https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=DeepGCNLayer#torch_geometric.nn.models.DeepGCNLayer
    """

    def __init__(self, conv=None, norm=None, act=None, block='res+',
                 dropout=0., ):
        super().__init__()

        self.conv = conv
        self.norm = nn.ModuleList(norm) if norm is not None else None
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'plain']  # 'res', 'dense',
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, *args, **kwargs):
        """"""
        args = list(args)
        x = args.pop(0)
        x_right = args.pop(0)
        h = x
        h_right = x_right
        if self.norm is not None:
            h = self.norm[0](h)
            h_right = self.norm[1](h_right)
        if self.act is not None:
            h = self.act(h)
            h_right = self.act(h_right)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h_right = F.dropout(h_right, p=self.dropout, training=self.training)
        h, h_right = self.conv(h, h_right, *args, **kwargs)
        if self.block == 'res+':
            return (x + h, x_right + h_right)
        else:
            return h, h_right

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'


class GCNDeeper(GCNBase):
    def __init__(self, p, q, hids=128, depth=50, *args, **kwargs):
        super(GCNDeeper, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvTwoDirection(p, q, hids))
        depth -= 2
        for d in range(depth):
            # if d == depth - 1:
            #     conv = GraphConvTwoDirection(hids, hids, 3)
            # else:
            conv = GraphConvTwoDirection(hids, hids, hids)
            norm = LayerNorm(hids), LayerNorm(hids)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, dropout=.1,
                                 block='res+'  # if d != depth - 1 else 'plain'
                                 )
            self.layers.append(layer)
        self.layers.append(GraphConvTwoDirection(hids, hids, 3))

    def forward(self, batch):
        left, right, edge_index, = batch.x_s, batch.x_t, batch.edge_index,
        x_left, x_right = left, right
        for layer in self.layers:
            left, right = layer(left, right, edge_index)

        left, right = add_knowledge(left, right, x_left, x_right)
        # left, right = F.log_softmax(left, dim=1), F.log_softmax(right, dim=1)
        return left, right


class GENDeeper(GCNDeeper):
    def __init__(self, p, q, hids=128, depth=5, *args, **kwargs):
        super(GCNDeeper, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GENConvTwoDirection(p, q, hids))
        depth -= 2
        for d in range(depth):
            conv = GENConvTwoDirection(hids, hids, hids)
            norm = LayerNorm(hids), LayerNorm(hids)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, dropout=.1,
                                 block='res+'
                                 )
            self.layers.append(layer)
        self.layers.append(GENConvTwoDirection(hids, hids, 3))


if __name__ == '__main__':
    mdl = GCN_FC(8, 8)
    mdl = GENDeeper(8,8) 
