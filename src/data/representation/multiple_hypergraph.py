'''
Author: M.S
Date: 2023-12-12 09:22:04
LastEditors: magical-seven
LastEditTime: 2024-03-07 11:37:46
FilePath: \first_Model\first_Model\src\data\representation\multiple_hypergraph.py
Description: 

Copyright (c) 2024 by M.S, All Rights Reserved. 
'''
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from scipy import sparse

from ._custom_typing import hyperedge_type, layer_type
from .multi_hypergraph import MultiHypergraph
from ..data_conversion import hye_list_to_binary_incidence
from itertools import combinations


class multiple_hypergraph(MultiHypergraph):
    """
    多网络超图的表示
    由四个对象组织
    A 表示层内部超边的权重
    B 表示层内部超边的关联矩阵
    inter_layer_edge  表示层间连接
    """

    def __init__(self,
                 A: Dict[layer_type: np.ndarray],
                 B: Dict[layer_type:Union[np.ndarray, sparse.spmatrix]],
                 D_C: Dict,
                 inter_layer_edge_dic: Dict[Tuple:List[Tuple]],  # {():[(),()]...}
                 hye: Optional[Dict[layer_type, List[hyperedge_type]]],  # {layeri:[(),(),()]}
                 layerlist: List[int] = None,
                 layerNum: int = None,
                 input_check: bool = False,
                 ):
        """
        :param A: 层超边的权重，形式为{layeri : [A_i]}，i层数量为E的超边[A_i]shape为(E,)
        :param B: 层内关联矩阵，形式为{layeri: [B_i]}， i层超边数量为E,节点的数量为N的关联矩阵[B_i]shape为(N, E)
        :param D: 层超边内节点度矩阵，形式为{layeri:[D_i]}，i层超边数量为E，节点数量为N的超边内度矩阵[D_i]，shape为(N,E)
        :param inter_layer_edge: 层交叉边，形式为{(layer1,layer2): [(layer1_nodeid, layer2_nodeid)...]}  所有层号从0开始
        :param hye: 层超边字典，形式为{layeri: [(nodeid,nodeid)..]}
        :param layerlist: 层号列表
        :param layerNum: 层数量
        :param input_check: 输入一致性检查
        """
        self.A = A
        self.B = B
        self.inter_layer_egde_dic = inter_layer_edge_dic
        self.hye = hye
        self.BA = self.get_BA()

        self.D_C = D_C
        self.C, self.Degree = self.get_D_C(D_C)


        if layerlist is not None:
            self.layerlist = layerlist
        else:
            self.layerlist = self.get_layername_list()

        if layerNum is not None:
            self.layerNum = layerNum

        if input_check:
            pass

    @staticmethod
    def _check_inputs(
            A,
            B,
            layerlist,
    ):
        pass

    def get_D_C(self, D_C: dict):
        """
        整理度矩阵与常数值
        """
        DC = D_C.copy()
        temp_C = DC['C']
        del DC['C']
        temp_D = DC
        return temp_C, temp_D

    def get_repr(self) -> Any:
        return self.A, self.B, self.hye

    def get_inter_layer_name_list(self) -> List:
        """
        获取具有层间连接的列表
        """
        return list(self.inter_layer_egde_dic.keys())  # [():[[],[]]...]

    def get_node_num(self):
        """
            Returns

            -------

            获取每一层的节点数目
        """
        N = {}
        for i in self.layerlist:
            N[i] = max(max(h) for h in self.hye[i]) + 1
        return N

    def get_layername_list(self):
        """
        获取层列表
        """
        Lylist = list(self.A.keys())
        assert Lylist == list(self.hye.keys())
        return Lylist

    def get_layerNum(self):
        """
        获取层的数量
        """
        return len(self.layerlist)

    def creat_LayerLink_matrix(self):
        """
            创建层与层的连接矩阵:
            若层l1与l2存在层间连接，则layerlink_matrix[l1,l2]=1
        """
        lnum = self.get_layerNum()
        layerlink_matrix = np.zeros((lnum, lnum))
        inter_layer_name = self.get_inter_layer_name_list()
        for items in inter_layer_name:
            i, j = items
            if layerlink_matrix[i, j] == 0:
                layerlink_matrix[i][j] = 1
                layerlink_matrix[j][i] = 1

    def get_max_hye_dic(self, DLbool: bool = True) -> Union[Dict, List]:
        '''
        description:获取超边列表或字典，层名作为索引 
        param {bool} DLbool
        return {*}
        '''
        max_hye_size_dic = {}
        layerlist = self.get_layername_list()
        for layer in layerlist:
            max = 0
            for tup_temp in self.hye[layer]:
                if len(tup_temp) > max:
                    max = len(tup_temp)
            max_hye_size_dic[layer] = max
        if DLbool:
            return max_hye_size_dic
        else:
            return list[max_hye_size_dic]

    @classmethod
    def trans_interlayeredgelist_to_coomatrix(
            cls,
            inter_layer_edge_dic: Dict[Tuple, List],
    ):
        """
        将层间边的列表变为压缩矩阵
        """
        inter_layer_edge_matric = {}
        layernamelist = list(inter_layer_edge_dic.keys())
        for index in layernamelist:
            edge_list = inter_layer_edge_dic[index]
            row, col = zip(*edge_list)
            value = np.ones(len(row))
            interarray = sparse.coo_matrix((value, (row, col)))
            inter_layer_edge_matric[index] = interarray.tocsr()  # 变为行压缩
        return inter_layer_edge_matric

    @classmethod
    def get_linklayer(
            cls,
            interlayerlist: List[Tuple],
            layername: int,
    ) -> Tuple:
        """
        获取与layername连接的层名
        """
        temp = []
        for item in interlayerlist:
            if layername in item:
                for i in item:
                    if i != layername:
                        temp.append(i)
        return tuple(temp)

    def get_hye_length(self):
        pass

    def get_BA(self):
        '''
        Returns
        -------
        获取B*A的矩阵
        '''
        layers = self.B.keys()
        BA = {}
        if sparse.issparse(self.B[0]):
            for l in layers:
                B = self.B[l]
                BA[l] = B.multiply(self.A[l])
        else:
            for l in layers:
                B = self.B[l]
                BA[l] = B * self.A[l]

        return BA

    def get_hye_degrees(self):
        """
        Returns
        -------
        计算每一层的节点超边内度
        """
        N_dic = self.get_node_num()
        degree_dic = {}
        for layer in self.hye.keys():
            hye = self.hye[layer]
            B = self.B[layer].todense()
            sumB = B.sum(axis=0)
            hye_indices = np.where(sumB > 2)[0]

            E = len(hye)
            N = N_dic[layer]
            D = np.zeros((N, E), dtype=int)
            for e in hye_indices:
                for f, h_sub in enumerate(hye):
                    if len(h_sub) >= len(hye[e]):
                        continue
                    if set(h_sub).issubset(hye[e]):
                        D[h_sub, e] += 1
            D += B
            sp_D = sparse.csc_matrix(D)
            degree_dic[layer] = sp_D
        return degree_dic

    @classmethod
    def C_l(cls, B: sparse.spmatrix):
        """
        用于计算常数值
        Returns
        -------

        """
        tem = B.sum(axis=0)
        colsum = np.array(tem).flatten()
        node_set_num = multiple_hypergraph.get_node_set(B)
        C_l = (np.sum((colsum - 1) * colsum / 2)) / node_set_num
        mu = multiple_hypergraph.get_mu(B)
        C_l = C_l * mu
        return C_l

    @classmethod
    def get_mu(cls, B):
        N = B.shape[0]
        tem = B.sum(axis=0)
        n = np.array(tem).flatten()
        mu = N / np.sum(2 / ((n - 1) * n))
        return mu

    @classmethod
    def get_node_set(cls, B: sparse.spmatrix):
        B = B.tocsc()
        index_combination = set()
        for col in range(B.shape[1]):
            rows = B[:, [col]].nonzero()[0]
            col_combination = combinations(rows, 2)
            index_combination.update(col_combination)
        return len(index_combination)
