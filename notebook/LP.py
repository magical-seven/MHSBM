import os.path
import sys
sys.path.append(r'E:\dzq\MHSBM')
import time
import numpy as np
import pandas as pd
from poprogress import simple_progress
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import multiprocessing as mp
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.model._linear_ops import qf, bf, qf_and_sum, bf_and_sum
from src.file_load import *
from src.data.representation.multiple_hypergraph import *
from src.data.data_io import multiple_hyper_load
from src.data.representation.intra_Hypergraph import *
from src.data.representation.layer_inter_edge import *
from src.model.mulit_model import *
from src.file_load import get_meta, get_hye_list
import function_for_LP as fLP


# 实验2
def Get_arry(dic: Dict[Union[int, str], List]) -> Dict[int, np.ndarray]:
    """
        将列表或者元组变为数组

        Parameters

        ----------

        dic

        Returns

        -------

        列表转数组
    """
    arry_dic = {}

    for layer, values in dic.items():
        if type(layer) == str:
            layer = tuple(int(x) for x in layer.split(','))
        if type(values[0]) == tuple:
            arry_dic[layer] = np.asarray(values, dtype=object)
        else:
            arry_dic[layer] = np.asarray(values)
    return arry_dic

def Get_E_N(A_dic: Dict[int, np.ndarray], B_dic: Dict[int, coo_matrix]):
    """
    返回超边和节点的数量信息
    Parameters
    ----------
    A_dic
    B_dic

    Returns
    -------
    E_dic, N_dic(层-边数量字典，层-节点数量字典)
    """
    E_dic = {}
    N_dic = {}
    for layer, v in A_dic.items():
        if type(v) == list:
            v = np.asarray(v)
        E_dic[layer] = v.shape[0]
    for layer, v in B_dic.items():
        N_dic[layer] = v.shape[0]
    return E_dic, N_dic

def train_model(i, K, seed, mphy, em_rounds):
    mphymodel = MultiHyMMSBM(
        K=K,
        seed=seed + i,
        assortative=True,

    )
    mphymodel.fit(mpgh=mphy, n_iter=em_rounds)
    loglike = mphymodel.log_like(mphy)
    return loglike, mphymodel


def get_best_model(K: list, em_rounds: int, mphy: MultiHypergraph, seed: int = 0, training_rounds: int = 10):
    bestmodel = None
    bestloglike = float('-inf')

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(train_model, i, K, seed, mphy, em_rounds): i for i in range(training_rounds)}

        for future in as_completed(futures):
            i = futures[future]
            loglike, mphymodel = future.result()
            if loglike > bestloglike:
                bestloglike = loglike
                bestmodel = mphymodel
            # print(f"the seed{seed + i} has trained over! the loglike value is {loglike}.")

    return bestmodel


# 最大似然，选取最优模型
def getbestMmodel(
        K: list, em_rounds: int, mphy: MultiHypergraph, seed: int = 0, training_rounds: int = 10, assortative=True
):
    """
        Parameters

        ----------

        K,
        em_rounds,
        mphy,
        seed,
        training_rounds,

        Returns

        -------

        获取最佳模型
    """
    bestmodel = None
    bestloglike = float('-inf')
    for i in simple_progress(range(training_rounds)):
        mphymodel = MultiHyMMSBM(
            K=K,
            seed=seed + i,
            assortative=assortative,
        )
        mphymodel.fit(mpgh=mphy, n_iter=em_rounds)
        loglike = mphymodel.log_like(mphy)
        if loglike > bestloglike:
            bestloglike = loglike
            bestmodel = mphymodel
        # print(f"the seed{seed + i} has trained over! the loglikevalue is {loglike}.")
    return bestmodel



def train_and_evaluate_for_k(K, seedlist, em_rounds, train_mphy, training_rounds, test_hye_dic_arr, N_dic, model,
                             seedm: int = 20):
    """
    在不同种子进行选择超边以计算AUC

    INPUT
    -------
    K:模型参数K
    seedlist:用于计算auc的种子列表
    em_rounds:算法循环次数
    train_mphy: 需要训练的超图
    training_rounds:算法轮回次数
    test_hye_dic_arr:测试的超边数组字典
    N_dic: 层节点数量字典
    seedm:int=20,模型种子
    """
    results_for_k = []
    # model = getbestMmodel(K, em_rounds, train_mphy, seed=seedm, training_rounds=training_rounds)
    u = model.u
    w = model.w
    for seed in seedlist:  # 在不同的种子下计算不同的auc
        # rseed = seed + np.random.default_rng(seed).integers(500)
        rseed = seed
        hyetrain = Get_arry(train_mphy.hye)
        auc = fLP.calculate_M_AUC(test_hye_dic_arr, u, w, N_dic, mask=None, n_cmparisons=1000, testflag=True,
                                  rseed=rseed, train_hye=hyetrain)
        results_for_k.append(auc)
        print(f"****{seed}种子下的auc为：{auc}与K配置{K}*******")
    return results_for_k

def keep_D_hye(test_hye: dict, maxD):
    '''
    只保留不大于maxD的超边
    '''
    test_hye_D = {}
    for l, hye in test_hye.items():
        hyL = [len(e) for e in hye]
        hyL2 = [eid for eid, d in enumerate(hyL) if 2 <= d <= maxD]
        test_hye_D[l] = hye[hyL2]
    return test_hye_D


if __name__ == '__main__':

    """数据准备"""
    # 实验2
    folder_path = r"data/link_predict/HIGHforLP"
    # 训练集路径
    train_hye_name = r'/hye_train.txt'
    train_weight_name = r'/weight_train.txt'
    train_inter_edges_name = r'/inter_edge_train.txt'
    # 测试集路径
    test_hye_name = r'/hye_test.txt'
    test_weight_name = r'/weight_test.txt'
    test_inter_edges_name = r'/inter_edge_test.txt'

    train_HY_path = [folder_path + train_hye_name, folder_path + train_weight_name,
                     folder_path + train_inter_edges_name]

    # 初始化训练集的超图
    train_mphy = multiple_hyper_load(hye_file=train_HY_path[0], weight_file=train_HY_path[1],
                                     inter_file=train_HY_path[2])
    A = train_mphy.A
    B = train_mphy.B
    _, N_dic = Get_E_N(A, B)

    # 测试集层-超边（列表）字典
    test_meta = get_meta(folder_path + test_hye_name)
    test_hye_dic_lst = get_hye_list(test_meta)
    test_hye_dic_arr = Get_arry(test_hye_dic_lst)

    # 保留不大于D的超边
    D = 200
    test_hye_dic_arrD = keep_D_hye(test_hye_dic_arr, D)

    # 训练参数
    training_rounds = 10
    em_rounds = 100
    modelseed = 10
    assortative = True
    K = [16,18]
    K_str = '-'.join(map(str, K))
    if type(assortative)==list:
        st = ",".join(map(str, assortative))
        file_name = f'{em_rounds}_{K_str}_{modelseed}_{assortative}_model.joblib'
    else:
        file_name = f'{em_rounds}_{K_str}_{modelseed}_{assortative}_model.joblib'
    modelpath = folder_path + '/save_model/' + file_name

    # 模型训练
    if os.path.exists(modelpath):
        model = joblib.load(modelpath)
    else:
        model = getbestMmodel(K, em_rounds, train_mphy, seed=modelseed, training_rounds=training_rounds, assortative=assortative)
        joblib.dump(model, modelpath)
    u = model.u
    # u_0_sum = np.sum(u[0], axis=1)
    # ux = u[1][[13191, 62234, 67696, 94416, 99235]]


    w = model.w

    # 生成计算auc的种子列表以及其他auc相关参数
    seedx = 0
    rng = np.random.default_rng(seedx)
    # seedlist = rng.integers(low=0, high=500, size=10)
    seedlist = [425]
    n_cmparisons = 1000
    testflag = True

    """计算auc"""
    AUC = []
    AUC_lst = []
    for se in seedlist:
        hyetrain = Get_arry(train_mphy.hye)
        auc, auc_lst = fLP.calculate_M_AUC(
            test_hye_dic_arrD, u, w, N_dic,
            mask=None, n_cmparisons=n_cmparisons, testflag=testflag, rseed=se, train_hye=hyetrain, auc_layer=True)
        AUC.append(auc)
        AUC_lst.append(auc_lst)
        print(f"****{se}种子下的auc为：{auc}与K配置{K}*******")
        print(f"两层auc列表{auc_lst}")

    print(AUC)
    print(np.mean(AUC), "\t", np.var(AUC))