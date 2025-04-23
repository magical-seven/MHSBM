from typing import Union, Dict, Optional, List, Tuple
from scipy import sparse
import numpy as np

from src.data.representation._custom_typing import layer_type, hyperedge_type
from src.data.representation.multiple_hypergraph import multiple_hypergraph

class intra_hypergraph(multiple_hypergraph):
    def __init__(self,
                 A,
                 B,
                 D,
                 hye: Dict[int, list],
                 inter_layer_edge_dic: Dict[int, np.ndarray],
                 layername:int,
                 A_layer: np.ndarray = None,
                 B_layer:Union[np.ndarray, sparse.spmatrix] = None,
                 hye_layer: List[hyperedge_type] = None
        ):
        '''
        description: 层超图类
        param {*} A:继承父类的超边权重字典
        param {*} B:继承父类的关联矩阵字典
        param {Dict} hye:继承父类超边字典数组
        param {Dict} inter_layer_edge_dic:继承父类的层间连接字典，形式为{(layer1,layer2): [[layer1_nodeid, layer2_nodeid]...]}  所有层号从0开始
        param {int} layername:层名
        param {np} A_layer:该层的超边权重
        param {Union} B_layer:该层的关联矩阵，shape=(N,E)
        param {List} hye_layer:该层超边
        return {*}
        '''
        super(intra_hypergraph, self).__init__(A, B, D,inter_layer_edge_dic, hye)

        self.layername = layername

        if A_layer is not None:  # 超边权重，以数组形式存在
            self.A_layer = A_layer
        else:
            self.A_layer = A[layername]

        if B_layer is not None:   # 关联矩阵，以稀疏矩阵形式
            self.B_layer = B_layer
        else:
            self.B_layer = B[layername]

        if hye_layer is not None:  # 层内超边，形式为[(node,node,node),.....]
            self.hye_layer = hye_layer
        else:
            self.hye_layer = hye[layername]



    # 获取layer层带权重的关联矩阵
    def get_layer_incidence_matrix(self) -> Union[np.ndarray, sparse.base.spmatrix]:
        return self.B_layer

    def get_layer_hye_weights(self) -> np.ndarray:
        return self.A_layer

    # 获取layer层二进制关联矩阵
    def get_layer_binary_incidence_matrix(self) -> Union[np.ndarray, sparse.base.spmatrix]:
        return self.B_layer > 0

    # 获取layer层超边的数量
    def get_layer_hyperedgeE(self) -> int:
        return self.A_layer.shape[0]

    # 获取层超边节点数量
    def get_layer_hypernodeN(self) -> int:
        N1 = max(map(max, self.hye_layer)) + 1
        N2 = self.B_layer.shape[0]
        N = max(N1,N2)
        return N

    def get_layer_cmtyK(self):
        """
        获取层社区数
        """
        pass

    # 获取最大超边
    def get_max_hye(self)->int:
        max_hye_size = 0
        for tup_temp in self.hye_layer:
            if len(tup_temp) > max_hye_size:
                max_hye_size = len(tup_temp)
        return max_hye_size

    # 获取该层的名称
    def get_layer_name(self)->int:
        return self.layername

    def __iter__(self):
        return zip(self.hye_layer, self.A_layer)

