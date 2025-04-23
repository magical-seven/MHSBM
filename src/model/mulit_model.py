import os.path
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix, find
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from poprogress import simple_progress
from itertools import combinations

from src.data.representation.multiple_hypergraph import multiple_hypergraph
from src.data.representation.intra_Hypergraph import intra_hypergraph
from src.data.representation.layer_inter_edge import inlyedge
from ._linear_ops import *
from ..f1_score import *


class MultiHyMMSBM:
    """
    多网络随机块模型
    """
    def __init__(
            self,
            K: list = None,
            u: Dict[int, np.ndarray] = None,
            w: Dict[int, np.ndarray] = None,
            assortative: Union[bool,List[bool]] = False,
            inter_w: Dict[Tuple, np.ndarray] = None,
            kappa_fn: str = "binom+avg",
            max_hye_size: Dict[int, int] = None,
            u_prior: Dict[int, np.ndarray] = 0.0,
            w_prior: Dict[int, np.ndarray] = 0.0,
            seed: Optional[int] = None,
            C: List[List] = None,
    ):
        if K is not None and isinstance(K, dict):
            self.K = K
        else:
            self.K = K

        # 层间社区之间的相互作用密度，亲和矩阵
        if inter_w == None:
            self.inter_w = {}
        else:
            self.inter_w = inter_w


        # 隶属度矩阵字典
        if u == None:
            self.u = {}
        else:
            self.u = u
        # 层内亲和矩阵字典
        if w == None:
            self.w = {}
        else:
            self.w = w

        if type(assortative) == bool:
            self.assortative = assortative
            self.layer_assort_flag = False
        else:
            self.assortative = assortative
            self.layer_assort_flag = True

        if max_hye_size is not None:
            self.max_hye_size = max_hye_size  # 最大超边
        else:
            self.max_hye_size = None

        self.kappa_fn = kappa_fn

        # 先验
        self.u_prior = u_prior
        self.w_prior = w_prior


        # 随机种子
        self._rng = np.random.default_rng(seed)
        self.seed = seed

        # 是否训练完成
        self.trained = False

        # 获取超图的计算常数
        if C is not None:
            self._C = C
        else:
            self._C = None


    @property
    def N(self) -> Dict[int, np.ndarray]:
        """
        获取节点数量
        """
        N = {}
        if self.u is None:
            return None
        else:
            key = list(self.u.keys())
            for i in key:
                N[i] = self.u[i].shape[0]
        return N

    def C(
            self,
            layername: int,
            d: Union[str, int, np.ndarray] = "all",
            return_summands: bool = False
    ) -> Union[np.ndarray, float]:
        r'''
        description:
        用于计算似然函数中的常数项
        公式如下：
            \sum_{d=2}^D \binom{N-2}{d-2}/ kappa_d
            参数 d 是一个表示超边维度的值，可以是单个值、整数或 NumPy 数组。默认值为字符串 "all"，表示计算所有维度。
            参数 return_summands 是一个布尔值，指示是否返回每个维度的单独项

        param {*} layername: 层名

        param {Union} d

        param {bool} return_summands

        return {*}
        '''
        d_vals = self._dimensions_to_numpy(layername, d)
        if self.kappa_fn == "binom+avg":
            res = 2 / (d_vals * (d_vals - 1))
        else:
            raise NotImplementedError()

        if not return_summands:
            res = res.sum()
        return res

    def check_C(self, mpgh):
        layernamelist = mpgh.get_layername_list()  # 获取层列表

        if self._C is None and mpgh.C is None:
            Cl = []
            for layer in layernamelist:
                Cl.append(self.C_l(mpgh.B[layer]))
            self._C = Cl
            mpgh.C = Cl
        else:
            if self._C is None:
                self._C = mpgh.C

        assert self._C == mpgh.C
        pass

    def _check_Multihyper_para(self, mpgh) -> None:
        """
        检查超图的参数是否符合要求的数据结构
        """
        if isinstance(self.K, List):
            self.K = self.layer_Klist_to_dict(mpgh=mpgh)

        if self.max_hye_size is None:
            self.max_hye_size = mpgh.get_max_hye_dic()

    def fit(self, mpgh: multiple_hypergraph, n_iter: int = 10, MDbool: bool = True):
        # 检查超图的部分参数
        self._check_Multihyper_para(mpgh)

        # degree
        D_mpgh = mpgh.Degree  # 获取各层节点内超边度信息
        layernamelist = mpgh.get_layername_list()  # 获取层列表
        lyed = inlyedge(mpgh.A, mpgh.B, mpgh.D_C, mpgh.hye, mpgh.inter_layer_egde_dic)  # 实例化一个层间连接
        cross_edge_dic = lyed.S_dic  # 层间连接权重字典
        in_layerlist = lyed.get_inter_layer_name_list()  # 获取具有层间连接的列表[(lyer1,lyer2)....]

        # 初始化各层，层内矩阵
        for layer in layernamelist:
            hypergraph = intra_hypergraph(mpgh.A, mpgh.B, mpgh.D_C, mpgh.hye, mpgh.inter_layer_egde_dic, layer)
            self._init_intra_w(layer)
            self._init_layer_u(hypergraph, layer)
        # 初始化层间矩阵
        for in_layer in in_layerlist:
            self._init_inter_w(mpgh=mpgh, interlayername=in_layer)

        # 获取常数值
        self.check_C(mpgh)

        MDbool = MDbool

        # EM步骤 train
        for it in range(n_iter):

            # 中间变量
            ul_new = {}  # 层内更新的u
            # wl_new = {}  # 层内更新的w
            wll_new = {}

            # 层内u和w  的更新
            for layer in layernamelist:
                linkLN = lyed.get_linklayer(interlayerlist=in_layerlist, layername=layer)

                A = mpgh.A[layer]
                B = mpgh.B[layer]

                D = D_mpgh[layer]
                ul_new[layer] = self._update_u(layername=layer, linkLN=linkLN, inter_edge_dic=cross_edge_dic,
                                               intra_binary_incidence=B, D=D, hye_weights=A, MDbool=MDbool)
                pass

                # 行归一化

                ul_new_sum = ul_new[layer].sum(axis=1, keepdims=True)
                ul_new_normalized = ul_new[layer] / np.where(ul_new_sum > 0, ul_new_sum, 1e-8)
                self.u[layer] = ul_new_normalized

                # 层内w的更新
                self.w[layer] = self._update_w_intra(layer, B, D, A)

                # 层间w的更新
                for otherL in linkLN:
                    if layer > otherL:
                        LtoL = (otherL, layer)
                    else:
                        LtoL = (layer, otherL)
                    wll_new[LtoL] = self._update_w_inter(LtoL[0], LtoL[1], cross_edge_dic[LtoL])
                self.inter_w = wll_new

        self.trained = True

    def savecom(self, path, iter, Com: dict):
        '''
        保存每一次社区迭代结果
        Parameters
        ----------
        path
        iter
        Com
        Returns
        -------
        '''

        if iter == 0:
            with open(path, 'w', encoding='utf-8') as wf:
                for k, v in Com.items():
                    wf.writelines(str(iter) + "---" + str(k) + '\n')
                    for comid, nodels in enumerate(v):
                        wf.writelines(str(comid) + "  ")
                        wf.writelines(str(node) + "  " for node in nodels)
                        wf.writelines("\n")
                    wf.writelines("\n")
                wf.writelines("============\n")
        else:
            with open(path, 'a', encoding='utf-8') as wf:
                for k, v in Com.items():
                    wf.writelines(str(iter) + "---" + str(k) + '\n')
                    for comid, nodels in enumerate(v):
                        wf.writelines(str(comid) + "  ")
                        wf.writelines(str(node) + "  " for node in nodels)
                        wf.writelines("\n")
                    wf.writelines("\n")
                wf.writelines("===========\n")

    def save_matrices_to_excel(self, matrices, file_name_prefix, iteration):
        """
        在第一次迭代时创建Excel文件，在后续迭代中追加数据。
        """
        for key, matrix in matrices.items():
            file_name = f"dataAN/{file_name_prefix}_{key}.xlsx"
            df = pd.DataFrame(matrix)
            df['Iteration'] = iteration  # 添加一列来指示迭代次数

            if iteration == 0:
                # 第一次迭代，创建文件并写入数据
                df.to_excel(file_name, sheet_name='Data', index=False)
            else:
                # 为后续迭代数据前添加一个空行
                empty_row = pd.DataFrame({col: np.NaN for col in df.columns}, index=[0])
                df = pd.concat([empty_row, df], ignore_index=True)
                # 后续迭代，追加数据
                with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    book = writer.book
                    startrow = writer.sheets['Data'].max_row if 'Data' in writer.sheets else None
                    df.to_excel(writer, sheet_name='Data', index=False, header=startrow is None, startrow=startrow)

    def savelog(self, path, iter):
        """
        Parameters
        生成日志并保存
        ----------
        path:保存路径
        iter:迭代次数
        Returns
        -------
        """
        u_key = self.u.keys()
        w_key = self.w.keys()
        interw_key = self.inter_w.keys()
        if iter == 0:  # 第一次迭代
            with open(path, "w", encoding='utf-8') as wf:
                wf.writelines(str(iter) + "\n")
                for i in u_key:
                    wf.writelines("u{}\n".format(i))
                    np.savetxt(wf, self.u[i])
                for j in w_key:
                    wf.writelines("w{}\n".format(j))
                    np.savetxt(wf, self.w[j])
                for k in interw_key:
                    wf.writelines("interw{}\n".format(k))
                    np.savetxt(wf, self.inter_w[k])
                wf.writelines("========================\n")
        else:
            with open(path, "a", encoding='utf-8') as wf:
                wf.writelines(str(iter) + "\n")
                for i in u_key:
                    wf.writelines("u{}\n".format(i))
                    np.savetxt(wf, self.u[i])
                for j in w_key:
                    wf.writelines("w{}\n".format(j))
                    np.savetxt(wf, self.w[j])
                for k in interw_key:
                    wf.writelines("interw{}\n".format(k))
                    np.savetxt(wf, self.inter_w[k])
                wf.writelines("========================\n")

    def log_like(self, mpgh: multiple_hypergraph):

        layernamelist = mpgh.get_layername_list()

        # 层内
        intra_loglike = 0
        for layer in layernamelist:
            # C = self.C(layer)
            D = mpgh.Degree[layer]
            Cl = self._C[layer]
            u_intra = self.u[layer]
            w_intra = self.w[layer]
            A = mpgh.A[layer]
            B = mpgh.B[layer]

            if sparse.issparse(B) or sparse.issparse(D):
                B_D = B.multiply(D)
            else:
                B_D = B * D

            intra_first_add = bf_and_sum(u_intra, w_intra)
            intra_second_add = np.dot(A, np.log(self.get_intra_lambda(layer, B, B_D)))
            intra_addition = -intra_first_add * Cl + intra_second_add
            intra_loglike += intra_addition

        # 层间
        inter_loglike = 0
        lyed = inlyedge(mpgh.A, mpgh.B, mpgh.D_C, mpgh.hye, mpgh.inter_layer_egde_dic)  # 实例化一个层间连接
        inter_layerlist = lyed.get_inter_layer_name_list()  # 获取具有层间连接的列表[(lyer1,lyer2)....]
        inter_S_dic = lyed.S_dic  # 层间权重字典
        for ly_t_ly in inter_layerlist:

            S = inter_S_dic[ly_t_ly]  # 层间连接

            l1 = ly_t_ly[0]  # 层
            l2 = ly_t_ly[1]

            inter_lambda = self.get_inter_lambda(l1, l2, S)
            # inter_lambda[inter_lambda == 0] = 1e-10  # 去零

            inter_first_add = np.sum(inter_lambda.data)

            # 确保 S 和 inter_lambda 都是稀疏矩阵
            if sparse.issparse(S) and sparse.issparse(inter_lambda):
                # 直接在 inter_lambda 的非零数据上计算对数
                log_data = np.log(inter_lambda.data)

                # 创建一个新的与 inter_lambda 形状和结构相同的稀疏矩阵
                log_inter_lambda = sparse.csr_matrix((log_data, inter_lambda.indices, inter_lambda.indptr),
                                                     shape=inter_lambda.shape)

                # 使用 S 和 log_inter_lambda 的乘法
                tempadd = S.multiply(log_inter_lambda)
                inter_second_add = tempadd.sum()
            else:
                # 如果 inter_lambda 不是稀疏矩阵，直接计算对数并执行乘法和求和
                log_inter_lambda = np.log(inter_lambda)
                tempadd = S.multiply(log_inter_lambda) if sparse.issparse(S) else S * log_inter_lambda
                inter_second_add = tempadd.sum()

            inter_addition = -inter_first_add + inter_second_add
            inter_loglike += inter_addition

        loglike = inter_loglike +  intra_loglike

        return loglike

    def _update_w_inter(self, layer1name, layer2name, inter_edge):
        '''
        description: 更新层间亲和矩阵w
        param {*} layer1name
        param {*} layer2name
        param {*} inter_edge: 层间边
        return {*}
        '''
        u1 = self.u[layer1name]
        u2 = self.u[layer2name]
        inter_w = self.inter_w[(layer1name, layer2name)]
        N1, K1 = u1.shape
        N2, K2 = u2.shape

        inter_lambda = self.get_inter_lambda(layer1name=layer1name, layer2name=layer2name, inter_edge=inter_edge)

        G = inter_edge.copy()
        G.data /= inter_lambda.data
        # assert G.shape == (N1, N2)

        # Numerator
        numerator = u1.T @ G @ u2
        # assert numerator.shape == (K1, K2)

        # Denominator
        multiple1 = u1.sum(axis=0)[:, None]
        multiple2 = u2.sum(axis=0)[None, :]
        # assert multiple1.shape == (K1, 1)
        # assert multiple2.shape == (1, K2)
        denominator = multiple1 * multiple2
        # assert denominator.shape == (K1, K2)

        return (numerator / denominator) * inter_w

    def _update_u(
            self,
            layername: int,
            linkLN: Tuple,
            inter_edge_dic: Dict[tuple[int, Any], np.ndarray],
            intra_binary_incidence: Union[np.ndarray, sparse.spmatrix],
            D: sparse.spmatrix,
            hye_weights: np.ndarray,
            MDbool: bool = True
    ) -> np.ndarray:
        '''
        description: 分成4加数
        param {*} layername:需要更新的层名
        param {*} linkLN: 与layername连接的层名,由元组存储
        param {*} inter_edge_dic:层间交叉边,使用字典进行索引
        param {*} intra_binary_incidence:关联矩阵
        param {*} hye_weights:超边权重
        param {*} MDbool:是否训练多维超图
        return {*}:更新后的u^l
        '''
        # 获取层内更新下的分子第一个加数和分母的第一个加数
        Nfirst_add, Dfirst_add = self._update_u_intra(layername=layername,
                                                      intra_binary_incidence=intra_binary_incidence, D=D,
                                                      hye_weights=hye_weights)


        # 获取层间更新下的分子和与分母和
        if MDbool:
            Nsenc_addL, Dsenc_addL = [], []
            for layer2 in linkLN:
                Llayer = (layername, layer2)
                i, j = self._update_u_inter(layer1name=layername,
                                            layer2name=layer2,
                                            inter_edge=inter_edge_dic[Llayer],
                                            )

                Nsenc_addL.append(i)
                Dsenc_addL.append(j)
            Nsenc_add = sum(Nsenc_addL)
            Dsenc_add = sum(Dsenc_addL)
        # assert Nsenc_add.shape==Nfirst_add.shape
        else:
            Nsenc_add = 0
            Dsenc_add = 0
        # sumstime = time.time()

        numerator =  Nfirst_add +  Nsenc_add

        Cl = self._C[layername]
        denominator =  (Dfirst_add * Cl) +  Dsenc_add

        u_lnew = numerator / denominator

        return u_lnew

    def _update_u_inter(
            self,
            layer1name: int,
            layer2name: int,
            inter_edge: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        description: # 层与层之间u更新
        param {*} layer1name
        param {*} layer2name
        param {*} inter_egde: 两层之间的连接
        param {*} NDbool: 是否返回更新值
        return {*}
        '''
        u1 = self.u[layer1name]
        u2 = self.u[layer2name]

        # N1, K1 = u1.shape
        # N2, K2 = u2.shape

        # w = self.inter_w[(layer1name,layer2name)]

        assert layer1name != layer2name
        if layer1name < layer2name:
            w = self.inter_w[(layer1name, layer2name)]
        else:
            w = self.inter_w[(layer2name, layer1name)].T

        inter_lambda = self.get_inter_lambda(layer1name=layer1name, layer2name=layer2name, inter_edge=inter_edge)

        G = inter_edge.copy()
        G.data /= inter_lambda.data

        # 更新u1
        # Numerator_u1. 分子
        u2_w = u2 @ w.T
        numerator1 = (G @ u2_w) * u1  # shape=(N1,K1)

        # Denominator_u1. 分母
        first = u2.sum(axis=0)
        denominator1 = (w * first).sum(axis=1)  # shape=(K1,)
        # assert denominator1.shape == (K1,)

        return numerator1, denominator1  # 返回分子分母

    def _update_u_intra(
            self,
            layername: Union[int, str],
            intra_binary_incidence: Union[np.ndarray, sparse.spmatrix],
            D: sparse.spmatrix,
            hye_weights: np.ndarray,
    ):  # 层内u更新
        '''
        description: 对于参数u的 EM or MAP 的更新
        param {Union} layername
        param {Union} intra_binary_incidence :关联矩阵，shape=(N,E),B
        param {np} hye_weights :A
        return {*}
        '''
        u, w = self.u[layername], self.w[layername]

        E = len(hye_weights)  # 超边个数
        N = u.shape[0]  # 节点个数
        K = self.K[layername]  # 社区个数

        # 获取 节点的超边内度
        if sparse.issparse(intra_binary_incidence):
            B_D = intra_binary_incidence.multiply(D)
        else:
            B_D = intra_binary_incidence * D

        # Numerator.   # 分子，获取泊松参数以及s_e
        poisson_params, edge_sum = self.get_intra_lambda(
            layername=layername, intra_binary_matrix=intra_binary_incidence, B_D=B_D, return_edge_sum=True
        )

        multiplier = hye_weights / poisson_params
        # assert multiplier.shape == (E,)

        if sparse.issparse(B_D):
            weighting = B_D.multiply(multiplier[None, :])
            # assert sparse.issparse(weighting)
        else:
            weighting = B_D * multiplier[None, :]
        # assert weighting.shape == (N, E)

        first_addend = weighting @ edge_sum
        # assert first_addend.shape == (N, K)

        if sparse.issparse(weighting):
            temp = weighting.multiply(D)
            weighting_sum = np.asarray(temp.sum(axis=1)).reshape(-1, 1)
        else:
            temp = weighting * D
            weighting_sum = temp.sum(axis=1, keepdims=True)
        second_addend = weighting_sum * u
        # assert second_addend.shape == (N, K)

        numerator = u * np.matmul(first_addend - second_addend, w)

        u_sum = u.sum(axis=0)
        denominator = np.matmul(w, u_sum)[None, :] - np.matmul(u, w)

        return numerator, denominator

    def _update_w_intra(
            self,
            layername,
            intra_binary_incidence,
            D,
            hye_weights,
    ):  # 层内w的更新
        """
        EM or MAP updates for the affinity matrix w.
        层内w的更新
        """
        u, w = self.u[layername], self.w[layername]

        E = len(hye_weights)
        N = u.shape[0]
        K = self.K[layername]

        # 获取 节点的超边内度
        if sparse.issparse(intra_binary_incidence):
            B_D = intra_binary_incidence.multiply(D)
        else:
            B_D = intra_binary_incidence * D

        poisson_params, edge_sum = self.get_intra_lambda(
            layername, intra_binary_incidence, return_edge_sum=True, B_D=B_D
        )

        multiplier = hye_weights / poisson_params
        # assert multiplier.shape == (E,)

        # Numerator : first addend s_ea * s_eb .
        first_addend = np.matmul(edge_sum.T, edge_sum * multiplier[:, None])
        # assert first_addend.shape == (K, K)

        # Numerator: second addend u_ia * u_ib .
        if sparse.issparse(B_D):
            B_D2 = B_D.multiply(B_D)
        else:
            B_D2 = B_D * B_D
        if sparse.issparse(B_D2):
            weighting = B_D2.multiply(multiplier[None, :]).sum(axis=1)
            weighting = np.asarray(weighting).reshape(-1)
        else:
            weighting = (B_D2 * multiplier[None, :]).sum(axis=1)
        # assert weighting.shape == (N,)

        second_addend = np.matmul(u.T, u * weighting[:, None])
        # assert second_addend.shape == (K, K)

        numerator = 0.5 * w * (first_addend - second_addend)

        # Denominator.
        Cl = self._C[layername]
        u_sum = u.sum(axis=0)
        denominator = 0.5 * (np.outer(u_sum, u_sum) - np.matmul(u.T, u))
        # assert denominator.shape == (K, K)

        return numerator / ((denominator + self.w_prior) * Cl)

    def get_inter_lambda(
            self,
            layer1name: int,
            layer2name: int,
            inter_edge: Union[np.ndarray, sparse.spmatrix],
    ):
        r"""
        :param layer1name: 层名1
        :param layer2name: 层名2
        ..math::
        \lambda_{ij}^{m_a m_b} = (u_{i}^{m_a})^{T}~w^{m_a m_b}~u_{j}^{m_b}
        =\sum_{k \in m_a, q \in m_b}u_{i k}^{m_a}~w_{k q}^{m_a m_b}~u_{j q}^{m_b}
        description:
        获取层与层之间的泊松参数
        """
        assert layer1name != layer2name
        u_l1 = self.u[layer1name]  # shape=(N1,K1)
        u_l2 = self.u[layer2name]  # shape=(N2,K2)

        N1 = u_l1.shape[0]
        N2 = u_l2.shape[0]

        if layer1name < layer2name:
            w = self.inter_w[(layer1name, layer2name)]

        else:
            w = self.inter_w[(layer2name, layer1name)].T

        # 寻找 inter_edge 中的非零元素位置
        rows, cols = inter_edge.nonzero()

        # 预计算乘积
        u_l1_w = u_l1 @ w

        values = (u_l1_w[rows] * u_l2[cols]).sum(axis=1)

        # 创建 COO 稀疏矩阵
        poisson_inter_coo = coo_matrix((values.ravel(), (rows, cols)), shape=(N1, N2))

        # 转换为 CSR
        poisson_inter = poisson_inter_coo.tocsr()

        # assert inter_poission_param.shape == (N1,N2)
        return poisson_inter

    def get_intra_lambda(
            self,
            layername,
            intra_binary_matrix: np.ndarray,
            B_D: Union[np.ndarray, sparse.spmatrix],
            return_edge_sum: bool = False):
        """
        获取层内的泊松参数
        :param layername:层名
        :param mpgh:多维超图
        :return:返回泊松参数，未归一化
        """
        u = self.u[layername]
        w = self.w[layername]
        E = intra_binary_matrix.shape[1]  # 超边数量
        K = self.K[layername]  # 群数量

        # 第一个加数，对于每一个超边 e: s_e^T w s_e .
        edge_sum = self._edge_sum(layername, B_D)  # 求s_e
        assert edge_sum.shape == (E, K)

        first_addend = qf(edge_sum, w)
        assert first_addend.shape == (E,)

        # 加数2: sum_{i \in e} u_i^T w u_i = \sum_{i \in e}u_{i}^{T} w u_{i}.
        if sparse.issparse(B_D):
            second_addend = (B_D.multiply(B_D)).T @ qf(u, w)
        else:
            second_addend = (B_D * B_D).T @ qf(u, w)
        assert second_addend.shape == (E,)

        poisson_params = 0.5 * (first_addend - second_addend)

        if return_edge_sum:
            return poisson_params, edge_sum  # 返回泊松参数和s_e
        else:
            return poisson_params

    def _edge_sum(
            self, layername, binary_incidence: Union[np.ndarray, sparse.spmatrix]
    ) -> np.ndarray:
        r"""
        :param layername: 层名
        :param binary_incidence: 关联矩阵
        :return : 返回s_e
        ..math::
            s_e = \sum_{i \in e}u_{i} = \sum_{i \in V} u_i@B
        """
        return binary_incidence.T @ self.u[layername]

    def get_layer_K(
            self,
            mpgh: multiple_hypergraph,
            layername: Union[str, int]
    ):
        """
        :param mpgh:多维超图
        :return: 获取层群落个数
        """
        if isinstance(self.K, List):
            self.layer_Klist_to_dict(mpgh=mpgh)
        return self.K[layername]

    def _init_inter_w(self, mpgh: multiple_hypergraph, interlayername: Tuple) -> None:
        """
        初始化层间社区亲和矩阵w
        :param mpgh: 多维超图
        :param interlayername: 层与层名，形式为（layer1name, layer2name）
        :return:
        """
        rng = self._rng  # self._rng = np.random.default_rng(seed)
        if isinstance(self.K, List):  # 将K变为字典类型 K存储每一层的社区数量，形式为{layer_name:community_num}
            self.layer_Klist_to_dict(mpgh=mpgh)
        if not isinstance(interlayername, Tuple):
            raise TypeError("类型错误，应为Tuple类型，你输入的类型为:", type(interlayername))

        # internamelist = lyed.S_dic  # 获取超图的层与层名列表 [(layer1name, layer2name)]
        internamelist = mpgh.get_inter_layer_name_list()  # 获取超图的层与层名列表 [(layer1name, layer2name)]

        if interlayername not in internamelist and interlayername[::-1] not in internamelist:
            raise ValueError("提供的层间边不存在", interlayername)

        K_layer1 = self.K[interlayername[0]]
        K_layer2 = self.K[interlayername[1]]
        shape = (K_layer1, K_layer2)
        self.inter_w[interlayername] = rng.random(shape).astype(np.float32)

    def _init_intra_w(self, layer: int):
        '''
        description:无视先验，直接进行初始化，随机数组，没有任何其他分布
        param {layer} layer:层名
        return {*}
        '''
        # 判断是否每一层以不同的分布方式
        layer_assort_flag = self.layer_assort_flag
        if layer_assort_flag:
            assortative = self.assortative[layer]
        else:
            assortative = self.assortative

        K_layer = self.K[layer]
        rng = self._rng

        w_layer = rng.random((K_layer, K_layer))
        self.w[layer] = np.triu(w_layer, 0) + np.triu(w_layer, 1).T  # 形成一个对角矩阵
        if assortative:  # 如果是同质
            self.w[layer] = np.diag(np.diag(self.w[layer]))  # 取对角线

    def _init_layer_w(self, layer: int):
        '''
        description: 初始化层内w
        param {layer} layer:层名
        return {*}
        '''
        layername = layer
        K_layer = self.K[layername]
        rng = self._rng

        # if self.w_prior == 0.1

        if isinstance(self.w_prior[layername], float) and self.w_prior[layername] == 0.0:  # 先验等于0
            w_layer = rng.random((K_layer, K_layer))
            self.w[layername] = np.triu(w_layer, 0) + np.triu(w_layer, 1).T
            if self.assortative:
                self.w[layername] = np.diag(np.diag(self.w[layername]))  # 取对角线
        else:  # w先验不为0
            if self.assortative:  # 同质
                if isinstance(self.w_prior[layername], float):  # 有先验，先验为浮点型
                    prior_mean = np.ones(K_layer) / self.w_prior[layername]
                else:  # 有先验，同质，数组
                    prior_mean = 1 / np.diag(self.w_prior[layername])
                self.w[layername] = np.diag(rng.exponential(prior_mean))

            else:  # 非同质
                if isinstance(self.w_prior[layername], float):
                    prior_mean = np.ones((K_layer, K_layer)) / self.w_prior[layername]
                else:
                    prior_mean = 1 / self.w_prior[layername]
                w_layer = rng.exponential(prior_mean)
                self.w[layername] = np.triu(w_layer, 0) + np.triu(w_layer, 1).T

    def _init_layer_u(self, intrahypergraph: intra_hypergraph, layer):
        '''
        description: 初始化层内的u
        param {*} self
        param {intra_hypergraph} intrahypergraph
        return {*}
        '''
        N_layer = intrahypergraph.get_layer_hypernodeN()
        K_layer = self.K[layer]
        rng = self._rng
        # 创建服从指数分布的初始的u
        self.u[layer] = rng.random((N_layer, K_layer))
        """
        if isinstance(self.u_prior, float) and self.u_prior == 0.0:
            self.u[layername] = rng.random((N_layer, K_layer))
        else:
            if isinstance(self.u_prior, np.ndarray):
                self.u[layername] = rng.exponential(1 / self.u_prior[layername])
            else:
                self.u[layername] = rng.exponential(1 / self.u_prior[layername], size=(N_layer, K_layer))
        """

    def layer_Klist_to_dict(self, mpgh: multiple_hypergraph) -> Dict:
        # 将K列表变为字典类型
        layername = mpgh.get_layername_list()
        if isinstance(self.K, List):
            temp = dict(zip(layername, self.K))
            Kdic = temp
            return Kdic
        else:
            return self.K

    def _dimensions_to_numpy(
            self, layername, d: Union[str, int, np.ndarray] = "all"
    ) -> np.ndarray:

        """
        返回某一层中的超边的维度
        简便方法，取某一层可能的超边缘维度的一些允许值d，并以数组格式返回。
        如果d已经是numpy数组，则返回它。
        如果d是整数，则将其包装在数组中。
        如果d是字符串"all"，则返回包含所有可能维度的数组，由self.max_hye_size指定。
        超边维度是指超边连接节点的数量
        超边维度就是一个超边中的节点数量
        """
        if isinstance(d, str) and d == "all" and self.max_hye_size[layername] is None:
            raise ValueError(
                "self.max_hye_size has not been specified. "
                "Either specify it or given d as input."
            )
        elif isinstance(d, str) and d == "all":
            # 生成一个一维数组,类型为(2,3,4...,self.max_hye_size[layername])
            d_vals = np.arange(2, self.max_hye_size[layername] + 1)

        elif isinstance(d, str):
            raise ValueError('Only string value for d is "all"')
        elif isinstance(d, int):
            d_vals = np.array([d])
        else:
            d_vals = d

        return d_vals

    @classmethod
    def C_l(cls, B: sparse.spmatrix):
        """
        用于计算常数值
        Returns
        -------

        """
        tem = B.sum(axis=0)
        colsum = np.array(tem).flatten()
        node_set_num = MultiHyMMSBM.get_node_set(B)
        C_l = (np.sum((colsum - 1) * colsum / 2)) / node_set_num
        mu = MultiHyMMSBM.get_mu(B)
        C_l = C_l * mu
        return C_l

    @classmethod
    def get_mu(cls, B):
        N = B.shape[0]
        tem = B.sum(axis=0)
        n = np.array(tem).flatten()
        mu = np.sum(2 / ((n - 1) * n)) / N
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

