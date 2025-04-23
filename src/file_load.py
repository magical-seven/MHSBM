from itertools import combinations

import numpy as np
from scipy import sparse, stats
import os
from .data.data_conversion import hye_list_to_binary_incidence
import pickle
import joblib
from poprogress import simple_progress
import math
from scipy.special import comb
from collections import defaultdict
from tqdm.auto import tqdm


# 保存对象
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# 加载pickle对象
def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 用于数据清洗
target_chars = ['==']  # 分层标识符或层间连接的表示符号
delete_table = str.maketrans({'=': None, '\n': None, "\ufeff":None})   # 清洗读取文件的无用字符




def get_meta(path):
    """
    获取元数据
    """
    # if not os.path.exists(path): #判断路径是否存在
    #     raise ValueError('路径错误', path)
    with open(path, 'r', encoding='utf-8') as rf:
        meta = rf.readlines()
    return meta


def is_numeric(input_str):
    """
    判断字符串类型
    """
    return input_str.isdigit()

def convert_to_number(value):
    """
    将字符串转变为整数或者浮点数
    """
    try:
        # 尝试将字符串转换为整数
        int_value = int(value)
        return int_value
    except ValueError:
        try:
            # 尝试将字符串转换为浮点数
            float_value = float(value)
            return float_value
        except ValueError:
            # 如果都失败，返回原始字符串
            return value

def get_indice(
        meta,
        split_char: list = None,
) -> list:
    """
    :param meta:
    :param split_char:
    :param clear_table:
    :return: 返回层索引
    """
    if split_char is None:
        split_char = target_chars
    indices = [index for index, element in enumerate(meta) if any(char in element for char in split_char)]
    return indices


def get_layerlist(
        hyemeta,
        chars=None,
        clear_table=None
):
    """
    获取层号列表
    """
    if clear_table is None:
        clear_table = delete_table
    if chars is None:
        chars = target_chars
    layerlist = []
    indices = [index for index, element in enumerate(hyemeta) if any(char in element for char in chars)]
    for index in indices:
        if is_numeric(hyemeta[index].translate(clear_table)):
            layerlist.append(int(hyemeta[index].translate(clear_table)))
        else:
            layerlist.append(hyemeta[index].translate(clear_table))
    return layerlist

def get_layer_nodeNum(hye: tuple) ->int:
    """求layer层最大节点序号作为layer层的节点个数"""
    maxN = 0
    for tu in hye:
        N = max(tu)
        if N > maxN:
            maxN = N
    return maxN + 1

def get_node_num(hye:dict):
    """
    Parameters

    ----------

    hye

    Returns

    -------

    获取每一层的节点个数
    """
    N = {}
    for layer in hye.keys():
        N[layer] = max(max(h) for h in hye[layer]) +1
    return N

# 获取权重列表 A
def get_A(
        weight_meta,
) -> dict:
    A = {}  # 权重列表
    indices = get_indice(weight_meta, target_chars)
    layerlist = get_layerlist(weight_meta)
    for i, item in enumerate(weight_meta):
        if i in indices:
            layerflag = layerlist[indices.index(i)]
        else:
            item = item.translate(delete_table)
            item = convert_to_number(item)   # 将其转变为数字类型
            # if isinstance()
            A.setdefault(layerflag, []).append(item)
    for key, value in A.items():
        A[key] = np.array(value)
    return A

def get_hye_list(hye_meta)->dict:
    """
    获取超边列表
    """
    hye = {}  # 超边链表
    indices = get_indice(hye_meta, target_chars)
    layerlist = get_layerlist(hyemeta=hye_meta)
    for i, item in enumerate(hye_meta):
        if i in indices:
            layerflag = layerlist[indices.index(i)]
        else:
            item = item.translate(delete_table)
            try:
                # item1 = item.split('\t')
                # item1 = tuple(int(x) for x in item1)
                item1 = item.split()
                item1 = tuple(int(x) for x in item1)
            except ValueError:
                item1 = item.split('   ')
                item1 = tuple(int(x) for x in item1)
            hye.setdefault(layerflag, []).append(item1)
    return hye

def get_B(hye, A) ->dict:
    """
    获取关联矩阵
    """
    B = {}  # 关联矩阵
    for key, value in hye.items():
        E = len(hye[key])
        N = get_layer_nodeNum(value)
        shape = (N, E)
        B[key] = hye_list_to_binary_incidence(value, shape=shape) * A[key]
        # B[key] = hye_list_to_binary_incidence(value, shape=shape)
    return B

def get_hye_Degree(B, hyed:dict) ->dict:
    """
    Returns
    -------
    获取超边内节点的度
    """
    # N_dic = get_node_num(hyed)
    degree_dic = {}
    for layer in hyed.keys():
        print('加载第{}的超边节点度'.format(layer))
        hye = hyed[layer]
        hye_size = [len(e) for e in hye]
        hye_size_array = np.array(hye_size)
        try:
            Bl = B[layer].todense()
        except:
            B[layer] = B[layer].astype(np.uint8)
            Bl = B[layer].todense()
        sumB = Bl.sum(axis=0)
        hye_indices = np.where(sumB > 2)[0]

        E = len(hye)
        N = Bl.shape[0]
        D = np.zeros((N,E))
        for e in simple_progress(hye_indices):
            for f, h_sub in enumerate(hye):
                if len(h_sub) >= len(hye[e]):
                    continue
                if set(h_sub).issubset(hye[e]):
                    D[h_sub,e] += 1
        D += Bl

        # D[D != 0] = 1
        try:
            normalizedD = D / D.sum(axis=0)
            D = normalizedD * hye_size_array
            sp_D = sparse.csc_matrix(D)
        except:
            D = sparse.csc_matrix(D)
            col_sums = D.sum(axis=0)
            normalizedD = D.multiply(1.0 / col_sums.A)
            size_diag = sparse.diags(hye_size_array)
            sp_D = normalizedD.dot(size_diag)


        # sp_D = sparse.csc_matrix(normalized_D)
        # sp_D = sparse.csc_matrix(D)
        degree_dic[layer] = sp_D
    return degree_dic

def get_inter_hye_degree(B, hyed: dict) -> dict:
    degree_dic = {}
    for layer in hyed.keys():
        print(f'加载第{layer}的超边节点度')
        hye = hyed[layer]
        hye_sizes = np.array([len(e) for e in hye], dtype=np.float64)
        
        try:
            Bl = B[layer]
        except KeyError:
            print(f"层 {layer} 不存在!")
            continue

        # 预计算超边哈希和子超边关系
        hye_sets = {idx: frozenset(e) for idx, e in enumerate(hye)}
        hye_children = defaultdict(list)
        sumB = Bl.sum(axis=0)
        hye_indices = np.where(sumB > 2)[0].tolist()

        with tqdm(hye_indices, desc=f"层 {layer} 预计算子超边",leave=False) as pbar:
        # for parent_idx in hye_indices:
            for parent_idx in pbar:
                parent_set = hye_sets[parent_idx]
                for child_idx, child_set in hye_sets.items():
                    try:
                        if child_set.issubset(parent_set) and len(child_set) < len(parent_set):
                            hye_children[parent_idx].append(child_idx)
                    except:
                        print(child_idx)
                        print(child_set)
                # pbar.set_postfix(f"子超边数量: {len(hye_children)}")

        # 批量构建 D 矩阵
        N, E = Bl.shape[0], len(hye)
        rows, cols, data = [], [], []
        with tqdm(hye_indices, desc=f"层 {layer} 构建 D 矩阵",leave=False) as pbar:
        # for e in hye_indices:
            for e in pbar:
            # for e in hye_indices:
                children = hye_children[e]
                for child_idx in children:
                    h_sub = hye[child_idx]
                    rows.extend(h_sub)
                    cols.extend([e] * len(h_sub))
                    data.extend([1] * len(h_sub))
                pbar.set_postfix({"子超边数量": len(children)})
        
        D = sparse.coo_matrix((data, (rows, cols)), shape=(N, E)).tocsc()
        D += Bl

        # 归一化
        col_sums = D.sum(axis=0).A.flatten()
        col_sums_nonzero = np.where(col_sums == 0, 1, col_sums)
        normalizedD = D.multiply(1.0 / col_sums_nonzero)
        size_diag = sparse.diags(hye_sizes)
        sp_D = normalizedD.dot(size_diag).tocsc()

        degree_dic[layer] = sp_D
        del rows, cols, data, hye_sets, hye_children  # 清理临时变量

    return degree_dic

def get_hye_De_pkl(path):
    try:
        with open(path, 'rb') as rf:
            degree_dic = pickle.load(rf)
    except:
        degree_dic = joblib.load(path)
    if 'C' not in degree_dic:
        raise KeyError(f"键 'C' 不存在于字典中")
    return degree_dic



def get_inter_layer_namelist(inter_edge_meta : dict) -> list:
    layerindices = get_indice(inter_edge_meta, split_char=['=='])
    # 解析存在相交边的层名称
    inter_layer_namelist = []
    for index in layerindices:
        item = inter_edge_meta[index]
        item = item.translate(delete_table)
        temp = [item[0], item[-1]]
        inter_layer_namelist.append(tuple([convert_to_number(x) for x in temp]))
    return inter_layer_namelist

def get_layer_dic(inter_edge_meta: dict) -> dict:
    """
        获取层次交叉边

        Parameters

        ----------

        inter_edge_meta

        Returns

        -------

    """
    # 解析层交叉边
    inter_layer_dic = {}

    layerindices = get_indice(inter_edge_meta, split_char=['=='])
    inter_layer_namelist = get_inter_layer_namelist(inter_edge_meta)
    for i, item in enumerate(inter_edge_meta):
        temp = item.translate(delete_table)
        if len(temp) == 0:
            continue
        if i in layerindices:
            flaglayer = inter_layer_namelist[layerindices.index(i)]
        else:
            try:
                # item1 = item.split('\t')
                # item1 = [int(x) for x in item1]
                item1 = item.split()
                item1 = tuple(int(x) for x in item1)
            except ValueError:
                item1 = item.split('   ')
                item1 = tuple(int(x) for x in item1)
            inter_layer_dic.setdefault(flaglayer, []).append(item1)
    return inter_layer_dic

def save_D_C(B:dict, path:str, degree_dic:dict, linkpredbool:bool=False, fold:str=None)->list:
    '''
    用于计算常数值，形成列表形式
    '''
    C = []
    for layer, Bl in B.items():
        C.append(get_C_l(Bl))

    degree_dic['C'] = C
    if linkpredbool:
        savepath = os.path.dirname(path) + '/degree_C_{}.pkl'.format(fold)
    else:
        savepath = os.path.dirname(path) + '/degree_C.pkl'
    with open(savepath, 'wb') as wf:
        pickle.dump(degree_dic, wf, protocol=pickle.HIGHEST_PROTOCOL)
    return C


def get_C_l(B: sparse.spmatrix):
    """
    用于计算常数值

    Returns
    -------

    """
    N = B.shape[0]
    hye_degree = B.sum(axis=0)  # 计算每一条超边的度
    colsum = np.array(hye_degree).flatten()
    mutiply = np.sum((colsum - 1) * colsum / 2)
    node_set_num = get_node_set(B)  # 获取每个超边里可能的节点二元互动数量
    first_add = 1 / node_set_num
    second_add = 2 / N*(N-1)
    C_l = mutiply * (first_add+second_add)
    return C_l


def get_node_set(B: sparse.spmatrix):
    '''
    计算所有超边里面可能的二元交互数量
    '''
    B = B.tocsc()
    index_combination = set()
    # for col in range(B.shape[1]):
    for col in simple_progress(range(B.shape[1])):
        rows = B[:, [col]].nonzero()[0]
        col_combination = combinations(rows, 2)
        index_combination.update(col_combination)
    return len(index_combination)


def check_pkl_file(path:str, filename:str = 'degree_C.pkl', linkpredbool:bool=False, fold:str=None):
    '''
    检查加载文件是否存在
    '''
    directory = os.path.dirname(path)
    if linkpredbool:
        filepath = directory + "/degree_C_{}.pkl".format(fold)
    else:
        filepath = directory + "/{}".format(filename)
    if os.path.exists(filepath):
        return filepath
    else:
        return None




