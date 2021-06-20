import numpy as np
import torch


def query_topk(node, x, M, k):
    '''
    :param node: SOM node of BxCxM tensor
    :param x: input data BxCxN tensor
    :param M: number of SOM nodes
    :param k: topk
    :return: mask: Nxnode_num
    '''
    # ensure x, and other stored tensors are in the same device
    device = x.device
    node = node.to(x.device)
    node_idx_list = torch.from_numpy(np.arange(M).astype(np.int64)).to(device)  # node_num LongTensor

    # expand as BxCxNxnode_num
    node = node.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2), M)
    x_expanded = x.unsqueeze(3).expand_as(node)

    # calcuate difference between x and each node
    diff = x_expanded - node  # BxCxNxnode_num
    diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

    # find the nearest neighbor
    _, min_idx = torch.topk(diff_norm, k=k, dim=2, largest=False, sorted=False)  # BxNxk
    min_idx_expanded = min_idx.unsqueeze(2).expand(min_idx.size()[0], min_idx.size()[1], M, k)  # BxNxnode_numxk

    node_idx_list = node_idx_list.unsqueeze(0).unsqueeze(0).unsqueeze(3).expand_as(
        min_idx_expanded).long()  # BxNxnode_numxk
    mask = torch.eq(min_idx_expanded, node_idx_list).int()  # BxNxnode_numxk
    # mask = torch.sum(mask, dim=3)  # BxNxnode_num

    # debug
    B, N, M = mask.size()[0], mask.size()[1], mask.size()[2]
    mask = mask.permute(0, 2, 3, 1).contiguous().view(B, M, k*N).permute(0, 2, 1).contiguous()  # BxMxkxN -> BxMxkN -> BxkNxM
    min_idx = min_idx.permute(0, 2, 1).contiguous().view(B, k*N)

    mask_row_max, _ = torch.max(mask, dim=1)  # Bxnode_num, this indicates whether the node has nearby x

    return mask, mask_row_max, min_idx

