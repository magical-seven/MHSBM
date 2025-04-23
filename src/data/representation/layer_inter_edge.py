r'''
Author: M.S
Date: 2024-03-05 20:30:42
LastEditors: magical-seven
LastEditTime: 2024-03-05 20:31:01
FilePath: \first_Model\first_Model\src\data\representation\layer_inter_edge.py
Description: 
用于描述超图与超图之间的边的函数类
Copyright (c) 2024 by M.S, All Rights Reserved. 
'''
from __future__ import annotations
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix

from src.data.representation.multiple_hypergraph import multiple_hypergraph


class inlyedge(multiple_hypergraph):
    def __init__(
            self,
            A,
            B,
            D_C,
            hye,
            inter_layer_edge_dic: Dict[Tuple:List[List]],  # {():[[],[]]...},
    ):
        super().__init__(A, B, D_C, inter_layer_edge_dic, hye)  # 继承父类的属性

        self.B = B
        self.layerlist = list(A.keys())

        self.inlyedge = inter_layer_edge_dic

        self.N_dic = self.get_N_dic()

        self.S_dic = self.trans_interlayeredgelist_to_coomatrix(inter_layer_edge_dic=inter_layer_edge_dic)

    def get_N_dic(self) -> dict:
        '''
        description: 获取节点数字典：{layer:node_num}
        return {*}
        '''
        N_dic = {}
        for layer, arr in self.B.items():
            N_dic[layer] = arr.shape[0]
        return N_dic

    def trans_interlayeredgelist_to_coomatrix(
            self,
            inter_layer_edge_dic: Dict[Tuple:List[List]],
    ) -> Dict:
        '''
        description: 将层间边的列表变为压缩矩阵，并且层名相反则取倒置
        param {Dict} inter_layer_edge_dic: 层间边，以字典形式存储的
        return {*}
        '''
        inter_layer_edge_matric = {}
        layernamelist = list(inter_layer_edge_dic.keys())
        for index in layernamelist:
            N1 = self.N_dic[index[0]]
            N2 = self.N_dic[index[1]]
            # Larr = np.zeros((N1, N2))
            edge_list = inter_layer_edge_dic[index]
            row, col = zip(*edge_list)
            interarray = coo_matrix((np.ones(len(row)),(row, col)), shape=(N1, N2))

            # Larr[(row, col)] = 1
            # interarray = sparse.coo_matrix(Larr)
            interarrayT = interarray.T
            inter_layer_edge_matric[index] = interarray.tocsr()  # 变为行压缩
            inter_layer_edge_matric[index[::-1]] = interarrayT.tocsr()
        return inter_layer_edge_matric

    def get_inter_layer_name_list(self) -> List:
        """
        获取具有层间连接的列表
        """
        return list(self.inter_layer_egde_dic.keys())

    def get_linklayer(
            self,
            interlayerlist: List[Tuple],
            layername: int,
    ) -> Tuple:
        '''
        description: 获取与layername连接的层名
        param {List} interlayerlist:具有连接的层间列表
        param {int} layername
        return {*}
        '''
        temp = []
        for item in interlayerlist:
            if layername in item:
                for i in item:
                    if i != layername:
                        temp.append(i)
        return tuple(temp)
