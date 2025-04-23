r'''
Author: M.S
Date: 2024-04-01 14:10:53
LastEditors: magical-seven
LastEditTime: 2024-04-01 14:12:23
FilePath: \first_Model\first_Model\src\calc_CS.py
Description: 计算CS指标
Copyright (c) 2024 by M.S, All Rights Reserved. 
'''
import numpy as np

def trans_com_matrix(Com:list)->np.ndarray:
    """
    将社区转变为隶属矩阵形式
    :param com: 社区信息
    :param Mshape: 社区数量
    :return: 隶属矩阵u
    """
    Mshape = len(Com)
    N = 0
    for i in Com:
        N += len(i)
    membership_matrix = np.zeros((N, Mshape))
    for comid, com in enumerate(Com):
        for nodeid in com:
            membership_matrix[nodeid][comid] = 1
    return membership_matrix


