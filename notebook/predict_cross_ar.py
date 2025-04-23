import sys
sys.path.append(r'E:\dzq\MHSBM')
from src.file_load import *
from src.data.representation.multiple_hypergraph import *
from src.data.data_io import multiple_hyper_load
from src.data.representation.layer_inter_edge import *
from src.model.mulit_model import *
from src.file_load import get_meta, get_hye_list
import function_for_LP as fLP

"""实验3 层间交叉边的链路预测"""
def Get_cross_array(dic: Dict[Union[int, str], List]) -> Dict[int, np.ndarray]:
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

# 最大似然，选取最优模型
def getbestMmodel(
        K: list, em_rounds: int, mphy: MultiHypergraph, seed: int = 0, training_rounds: int = 10, assortative:bool=True
):
    r"""
        Parameters

        ----------

        K:
            community of number
        em_rounds:
            the number for training
        mphy:
            the multiple hypergraph
        seed:
            the seed for initial
        training_rounds:
            the training
        assortative:
            True

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


if __name__ == '__main__':

    """准备训练集与测试集"""
    # 文件路径
    document = r'..\data\link_predict_inter\author_cite'
    hyepath = fr'{document}/hye.txt'
    weight_path = rf'{document}/weight.txt'

    train_inter_path = rf'{document}/inter_edge_train.txt'  # 训练集路径
    test_inter_path = rf'{document}/inter_edge_test.txt'  # 测试集路径

    train_meta = get_meta(train_inter_path)
    train_edge_dic_lst = get_layer_dic(train_meta)
    train_edge_arr = Get_cross_array(train_edge_dic_lst)  # 训练集边集

    test_meta = get_meta(test_inter_path)
    test_edge_dic_lst = get_layer_dic(test_meta)
    test_edge_arr = Get_cross_array(test_edge_dic_lst)  # 测试集边集

    """训练集模型的训练"""
    train_mpgh = multiple_hyper_load(hye_file=hyepath, weight_file=weight_path, inter_file=train_inter_path)  # 训练集多超图
    # 训练参数
    training_roundings = 5
    em_rounds = 65
    K = [10,10]
    modelseed = 15
    assortative = True
    K_str = '-'.join(map(str, K))
    filename = f'{em_rounds}_{K_str}_{modelseed}_{assortative}'

    # 训练保存模型
    if os.path.exists(rf"{document}/save_model/{filename}_model.joblib"):
        model = joblib.load(rf"{document}/save_model/{filename}_model.joblib")
    else:
        model = getbestMmodel(K=K, em_rounds=em_rounds, mphy=train_mpgh, seed=modelseed, training_rounds=training_roundings)
        joblib.dump(model, rf"{document}/save_model/{filename}_model.joblib")


    """计算auc"""
    linked_layer_name = train_mpgh.get_inter_layer_name_list()  # 获取连接层的层名关系
    # 设置相关参数
    n_comparions = 1000
    test_flag = True
    seed = 42
    rng = np.random.default_rng(seed)
    seedlst = rng.integers(0, 500, 10)
    # train =
    AUC = {}
    for l_to_l in linked_layer_name:
        u_fst = model.u[l_to_l[0]]
        u_sed = model.u[l_to_l[1]]
        w_inter = model.inter_w[l_to_l]
        N_lst = [u_fst.shape[0], u_sed.shape[0]]
        auclst = []
        for sd in seedlst:
            # 输入参数
            param = {
                'test_flag': True,
                'train': train_edge_arr[l_to_l],
                'test': test_edge_arr[l_to_l],
                'layertolayer': l_to_l,
                'N': N_lst,
                'seed': sd
            }
            auc = fLP.calculate_AUC_for_cross(u_fst, u_sed, w_inter, n_comparions, **param)
            auclst.append(auc)
            print(f"####{sd}种子下的auc为{auc}")
        AUC[l_to_l] = auclst

    for l_to_l, a in AUC.items():
        print(f'层{l_to_l[0]}与层{l_to_l[1]}之间交叉边的auc：')
        print(*a, sep='\t')
        print("mean and var:")
        print(np.mean(a), "\t",np.var(a))


