from typing import Any, Optional

import numpy as np
from scipy import sparse

from .representation.multiple_hypergraph import multiple_hypergraph
from ..file_load import *

def multiple_hyper_load(
        hye_file: str = '',
        weight_file: str = '',
        inter_file: str = '',
        degree_pickle: str = '',
        pickle_bool:bool=False
) -> multiple_hypergraph:
    """
    解析三个文件，超边文件，权重文件（超边对应的），内部交叉边文件
    :param hye_file: 超边文件路径
    :param weight_file: 权重文件
    :param inter_file: 内部交叉边
    :param degree_pickle: 存储超编度矩阵的文件
    :return: 多超图类
    """
    if (bool(hye_file) and bool(weight_file) and bool(inter_file)) is False:
        raise ValueError('Provided file invaild')
    if pickle_bool:
        pass
    if hye_file and weight_file and inter_file:
        hyemeta = get_meta(hye_file)
        weightmeta = get_meta(weight_file)
        interedgemeta = get_meta(inter_file)
        A = get_A(weight_meta=weightmeta)
        hye = get_hye_list(hye_meta=hyemeta)
        B = get_B(hye, A)
        interedge = get_layer_dic(inter_edge_meta=interedgemeta)
        DC_pickle = check_pkl_file(hye_file)
        if degree_pickle:
            D_C = get_hye_De_pkl(degree_pickle)
        elif DC_pickle is not None:
            D_C = get_hye_De_pkl(DC_pickle)
        else:
            D_C = get_inter_hye_degree(B, hyed=hye)
            D_C['C'] = save_D_C(B, path=hye_file, degree_dic=D_C)
        return multiple_hypergraph(A=A, B=B, inter_layer_edge_dic=interedge, hye=hye, D_C=D_C)
