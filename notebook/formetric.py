import numpy as np


def get_true_com(filepath):
    with open(filepath, "r", encoding='utf-8') as f:
        commeta = f.readlines()
        trueCom = {}
        for comid, com in enumerate(commeta):
            # temp = com.replace("\n", '')
            # nodeids = com.strip().split('\t')
            nodeids = com.strip().split()
            trueCom[comid] = [int(node) for node in nodeids]
    return trueCom

def comlist_to_matrix(h_true_pathcom):
    Htruecom = get_true_com(h_true_pathcom)  # 真实社区
    # 将真实社区表示为二维数组
    nodeN = sum(len(i) for i in Htruecom.values())
    comN = len(Htruecom.keys())
    U_true = np.zeros((nodeN, comN))
    for k, v in Htruecom.items():
        for id in v:
            U_true[id][k] = 1
    return U_true


def max_for_np(array, true_array):
    max_in_row = np.max(array, axis=1, keepdims=True)

    result = np.zeros_like(array)
    result[array == max_in_row] = 1

    temp = (true_array - result).astype(int)

    non_zero_rows = np.any(temp != 0, axis=1)
    row_indices = np.where(non_zero_rows)[0]

    return row_indices


def Get_cross_array(dic):
    arry_dic = {}

    for layer, values in dic.items():
        if type(layer) == str:
            layer = tuple(int(x) for x in layer.split(','))
        # arry_dic[layer] = np.array(values, dtype=object)
        cross_arr = np.empty(len(values), dtype=object)
        for i, e in enumerate(values):
            cross_arr[i] = e
        arry_dic[layer] = cross_arr
    return arry_dic