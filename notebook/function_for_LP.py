"""
    Functions used in the k-fold cross-validation procedure.
"""
import sys
import random

import numpy as np
from itertools import combinations
import os
import pickle
from typing import Dict, List
# sys.path.append('..')
from src.f1_score import get_hye_distribution, get_com


def sample_zero_n(hye, oneAUCG, N, rseed):
    """
        Sample zero entries.
        采样0条目

        INPUT
        -------
        hye : ndarray，超边矩阵
              Array of length E, containing the sets of hyperedges (as tuples).
        oneAUCG : list 包含存在超边的抽样列表
                  List containing samples of the existing hyperedges.
        N : int  节点数目
            Number of nodes.
        rseed : int  随机种子
                Random seed.

        OUTPUT
        -------
        sample_zero : list  返回含有不存在超边的列表
                      List containing non-existing hyperedges.
    """
    random.seed(int(rseed))
    # rng = np.random.RandomState(rseed)
    sample_zero = np.zeros_like(oneAUCG)
    nonzero_or_sampled = set(hye)
    for eid, e in enumerate(oneAUCG):
        d = len(e)  # of the same degree so we have balanced sets
        found_nnz = False
        while not found_nnz:
            t = tuple(sorted(random.sample(range(N), d)))
            if t not in nonzero_or_sampled:
                nonzero_or_sampled.add(t)
                found_nnz = True
                sample_zero[eid] = t

    return sample_zero





def prob_greater_zero_Poisson(lmbd):
    """
        P(A>0) for a Poisson with mean lmbd.
    """
    return 1.0 - np.exp(-lmbd)


def calculate_M_AUC(
        hye: Dict[int, np.ndarray],
        u: Dict[int, np.ndarray],
        w: Dict[int, np.ndarray],
        N: Dict[int, int],
        mask=None,
        n_cmparisons: int = 1000,
        rseed: int = 10,
        testflag: bool = False,
        train_hye: Dict[int, np.ndarray] = None,
        auc_layer:bool=False
):
    """

        Parameters

        ----------

        hye :
            多维超图的层-超边字典,测试集
        u :
            训练得到的u
        w :
            训练得到的w
        N :
            多维超图的层-节点字典
        mask:
            训练集与测试集的层-索引掩码
        n_cmparisons:
            选取对比的值
        rseed:
            随机种子
        testflag:
            测试标记
        train_hye:
            训练集超边
        Returns

        -------

        返回多维超图的每一层AUC的加和平均
    """

    rng = np.random.default_rng(rseed)
    seedlist = rng.integers(low=0, high=500, size=len(hye.keys()))
    layer_comparisons = n_cmparisons // len(hye.keys())
    AUC_lst = []
    for layer, hye in hye.items():
        if mask is None:
            if testflag:
                auc = calculate_AUC_layer(
                    hye, u[layer], w[layer], N[layer], mask, layer_comparisons, rseed, testflag, train_hye[layer])
            else:
                auc = calculate_AUC_layer(
                    hye, u[layer], w[layer], N[layer], mask, layer_comparisons, rseed)
        else:
            auc = calculate_AUC_layer(
                hye, u[layer], w[layer], N[layer], mask[layer], layer_comparisons, rseed)
        AUC_lst.append(auc)
    if auc_layer:
        return np.mean(AUC_lst), AUC_lst
    else:
        return np.mean(AUC_lst)


def calculate_AUC_layer(
        hye: np.ndarray, u: np.ndarray, w: np.ndarray, N: int, mask=None, n_comparisons: int = 100, rseed: int = 10,
        testflag: bool = False, train_hye: np.ndarray = None):
    """
        计算单层AUC

        Parameters

        ----------

        hye :
            超图的超边数组，测试集或训练集
        u :
            训练得到的某一层的u
        w :
            训练得到的某一层的w
        N :
            某一层超图的节点数
        mask:
            某一层训练集与测试集的索引掩码
        n_cmparisons:
            选取对比的值
        rseed:
            随机种子
        testflag:
            判定是否在进行五折实验， True表示不在五折实验

        Returns

        -------

        返回单层auc
    """
    # 设置种子
    rng = np.random.default_rng(rseed)
    # rseed += rng.integers(500)

    if mask is None:
        oneAUCG = rng.choice(hye, n_comparisons, replace=True)
    else:
        oneAUCG = rng.choice(hye[mask], n_comparisons, replace=True)


    if testflag:  # 如果是最后测试实验，那么
        assert train_hye is not None
        hye_all = np.concatenate((hye, train_hye), axis=0)
        train_hye_set = {n for h in train_hye for n in h}
    else:
        hye_all = hye
    zeroAUCG = sample_zero_n(hye_all, oneAUCG, N, rseed)

    if testflag:
        assert train_hye is not None
        oneD = get_D(train_hye, oneAUCG)
        zeroD = get_D(train_hye, zeroAUCG)
    else:
        oneD = get_D(hye[~mask], oneAUCG)
        zeroD = get_D(hye[~mask], zeroAUCG)

    maxD = max([len(e) for e in hye])

    R1 = []
    # count1=0
    for h, d in zip(oneAUCG, oneD):
        if len(h) > maxD:
            p_lambda = 0
        else:
            d_m = np.array(d).reshape(len(d), 1)
            # if set(h)& (train_hye_set):
            #     count1+=1
            u_d = u[np.array(h)] * d_m
            p_lambda = np.triu(u_d @ w @ u_d.T, 1).sum()
        R1.append(p_lambda)
    R1 = np.array(R1)

    R0 = []
    # count0 = 0
    for h, d in zip(zeroAUCG, zeroD):
        assert h not in set(hye_all)
        if len(h) > maxD:
            p_lambda = 0
        else:
            # if set(h)& (train_hye_set):
            #     count0+=1
            d_m = np.array(d).reshape(len(d), 1)
            u_d = u[np.array(h)] * d_m
            p_lambda = np.triu(u_d @ w @ u_d.T, 1).sum()

        R0.append(p_lambda)
    R0 = np.array(R0)

    assert R0.shape[0] == n_comparisons
    assert R1.shape[0] == n_comparisons

    n1 = (R1 > R0).sum()

    # print(R1>R0)

    equal = 0
    for r1, r0 in zip(R0, R1):
        if r1==0 and r0==0:
            equal += 1
    n_tie = (R1 == R0).sum()
    n_tie = n_tie - equal

    auc = (n1 + 0.5 * n_tie) / n_comparisons

    return auc


def shuffle_indices(n_samples, rng):
    """
        Shuffle the indices of the hyperedges.

        INPUT
        -------
        n_samples : int
                    Number of hyperedges.
        rng : RandomState
              Container for the Mersenne Twister pseudo-random number generator.

        OUTPUT
        -------
        indices : ndarray
                  Shuffled array with the indices of the hyperedges.
    """

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    return indices


def extract_mask_kfold(indices, fold=0, NFold=5):
    """
        Return the mask for selecting the held out set.

        INPUT
        -------
        indices : ndarray
                  Shuffled array with the indices of the hyperedges.
        fold : int
               Current fold.
        NFold : int
                Number of k-fold.

        OUTPUT
        -------
        mask : ndarray
               Mask for selecting the held out set.
    """

    n_samples = indices.shape[0]
    mask = np.ones(n_samples, dtype=bool)
    test = indices[fold * (n_samples // NFold): (fold + 1) * (n_samples // NFold)]
    mask[test] = 0

    return mask


def appear(ind, AUCG):
    # 检查超边是否已经出现过
    for i, e in enumerate(AUCG):
        if i >= ind:
            return 0
        elif e == AUCG[ind]:
            return i + 1


def get_D(hye_train, AUCG):
    """
        Parameters

        ----------

        hye_train:
            用于训练的超边
        AUCG:
            选择出来的呃超边

        Returns

        -------

        获取每一条选择的超边在训练的超边中的超边内度
    """
    # 获取超边内度
    hye_train = [sorted(h) for h in hye_train]
    hye_train_set = [set(h) for h in hye_train]
    D = []
    for i, e in enumerate(AUCG):
        d = {node: 1 for node in e}

        if len(e) <= 2:  # 如果超边小于等于2不存在子超边
            skeys = sorted(d.keys())
            values = [d[key] for key in skeys]
            D.append(values)
            continue

        ind = appear(i, AUCG)  # 搜索是否已经计算过该超边内度
        if ind:  # 如果存在，直接从已保存的值里，重复再加入列表D中
            D.append(D[ind - 1])
        else:  # 如果都不是，开始遍历hye_train
            for h in hye_train_set:
                if set(h).issubset(e):
                    for node in h:
                        d[node] += 1
            skeys = sorted(d.keys())
            values = [d[key] for key in skeys]
            D.append(values)
    return D

def sample_cross_zero_n(edge, oneAUCG, N_lst, seed):
    """
        Parameters

        ----------

        edge :
            训练集与测试集/验证集的交叉边
        oneAUCG :
            抽样的交叉边
        N_lst
            两层的节点数量
        seed
            抽样种子

        Returns

        -------

        返回一个抽样的不存在的交叉边

    """
    random.seed(int(seed))
    N_fst, N_sed = N_lst
    sample_zero = np.zeros_like(oneAUCG)
    nonzero_or_sampled = set(edge)
    for eid, e in enumerate(oneAUCG):
        found_nnz = False
        while not found_nnz:
            nodefst = random.choice(range(N_fst))
            nodesed = random.choice(range(N_sed))
            e = tuple((nodefst, nodesed))
            if e not in nonzero_or_sampled:
                nonzero_or_sampled.add(e)
                found_nnz = True
                try:
                    sample_zero[eid] = e
                except:
                    sample_zero[eid] = [e]


    return sample_zero

def calculate_AUC_for_cross(
        u_fst: np.ndarray,
        u_sed: np.ndarray,
        w_inter: np.ndarray,
        n_comparisons: int = 100,
        **kwargs
):
    """
        Parameters

        ----------

        u_fst:
            the first layer of u
        u_sed:
            the second layer of u
        w_inter:
            layer to layer of w
        n_comparisons:
            select number
        kwargs:
            if test_flag: train（训练集）,test（测试集）, layertolayer,N_lst(两层的节点数量), seed(choose the edge)

            else: mask（掩码）, edge（全部边集）, layertolayer,N_lst(两层的节点数量), seed(choose the edge)

        Returns

        -------

        计算层layerfirst 与 layersecond之间的层交叉边AUC
    """

    seed = kwargs['seed']
    rng = np.random.default_rng(seed)
    testflag = kwargs['test_flag']
    N_lst = kwargs['N']

    # 获取mask
    if not testflag:
        assert kwargs.get('mask') is not None
        mask = kwargs['mask']
    else:
        mask = None

    # 获取训练集和测试集/验证集
    if not testflag:
        inter_e = kwargs['edge']
        train_inter_e = inter_e[~mask][:,0]
        test_inter_e = inter_e[mask][:,0]
    else:
        train_inter_e = kwargs['train']
        test_inter_e = kwargs['test']

    oneAUCG = rng.choice(test_inter_e, n_comparisons, replace=True)  # 从测试集/验证集中抽取存在边

    # 抽取非存在边
    edge = np.concatenate((train_inter_e, test_inter_e), axis=0)
    zeroAUCG = sample_cross_zero_n(edge, oneAUCG, N_lst, seed)

    #
    R1 = []
    for e in oneAUCG:
        node_fst, node_sed = e
        p_inter_lambda = u_fst[node_fst] @ w_inter @ u_sed[node_sed].T
        R1.append(p_inter_lambda)
    R1 = np.array(R1)

    R0 = []
    for e in zeroAUCG:
        node_fst, node_sed = e
        p_inter_lambda = u_fst[node_fst] @ w_inter @ u_sed[node_sed].T
        R0.append(p_inter_lambda)
    R0 = np.array(R0)

    assert R0.shape[0] == n_comparisons
    assert R1.shape[0] == n_comparisons

    n1 = (R1 > R0).sum()
    n_tie = (R1 == R0).sum()
    auc = (n1 + 0.5 * n_tie) / float(n_comparisons)
    return auc


"""
以下为用于数据分析的函数
"""
