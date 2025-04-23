from typing import Union, Set, List, Tuple, Dict
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


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


def get_label(arr):
    label = []
    for row in arr:
        row = row.tolist()
        col = row.index(max(row))
        label.append(col)
    return label


def get_com(arr, dictB: bool = False) -> Union[Dict, List]:
    """
    通过隶属度矩阵推断社区分布
    Parameters
    ----------
    arr: 隶属度矩阵u

    Returns 社区字典或列表
    -------
    """
    if dictB:
        Com = {}  # 社区与节点字典
        com_num = arr.shape[1]  # 获取社区数

        for com in range(com_num):
            Com[com] = []
        for node, row in enumerate(arr):
            row = row.tolist()
            col = row.index(max(row))
            Com[col].append(node)
    else:
        Com = []  # 社区与节点字典
        com_num = arr.shape[1]  # 获取社区数

        for com in range(com_num):
            Com.append([])

        for node, row in enumerate(arr):
            row = row.tolist()
            col = row.index(max(row))
            Com[col].append(node)
    return Com


def calculate_multiclass_metrics(predictions, true_labels, num_classes):
    # Initialize dictionaries to store metrics for each class
    true_positives = {i: 0 for i in range(num_classes)}
    false_positives = {i: 0 for i in range(num_classes)}
    false_negatives = {i: 0 for i in range(num_classes)}

    # Calculate metrics for each class
    for pred, true in zip(predictions, true_labels):
        if pred == true:
            true_positives[true] += 1
        else:
            false_positives[pred] += 1
            false_negatives[true] += 1

    return true_positives, false_positives, false_negatives


def calculate_f1_score(true_positives, false_positives, false_negatives):
    f1_scores = {}
    for class_label in true_positives.keys():
        precision = true_positives[class_label] / (true_positives[class_label] + false_positives[class_label])
        recall = true_positives[class_label] / (true_positives[class_label] + false_negatives[class_label])

        # Handling cases where precision or recall is zero
        if precision == 0 or recall == 0:
            f1_scores[class_label] = 0
        else:
            f1_scores[class_label] = 2 * (precision * recall) / (precision + recall)

    # Calculate average F1 score
    avg_f1_score = sum(f1_scores.values()) / len(f1_scores)

    return avg_f1_score, f1_scores


def cal_f1(truelabel, predlabel, num_classes):
    true_positives, false_positives, false_negatives = calculate_multiclass_metrics(predlabel, truelabel, num_classes)
    avg_f1_score, f1_scores = calculate_f1_score(true_positives, false_positives, false_negatives)
    print("Average F1 Score for u1:", avg_f1_score)
    print("F1 Scores per Class for u1:", f1_scores)


def eval_scores(pred_comm: Union[List, Set],
                true_comm: Union[List, Set]) -> Tuple[float]:
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth community.
    """
    intersect = set(true_comm) & set(pred_comm)
    p = len(intersect) / len(pred_comm)
    r = len(intersect) / len(true_comm)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect))
    return round(p, 4), round(r, 4), round(f, 4), round(j, 4)


def get_node_label(com: dict):
    vL = list(com.values())
    # labelN = len(vL)
    # nodeN = sum(len(i) for i in vL)
    nodeN = len({node for cl in vL for node in cl})
    labelL = [0] * nodeN
    for k, v in com.items():
        for nodeid in v:
            labelL[nodeid] = k
    return labelL


def get_hye_distribution(hye: np.ndarray, com_path="", com: dict = None):
    """
    返回超边的在社区中的分布数量
    """
    if com is None:
        com = get_true_com(com_path)
    com_set = {}
    for k, v in com.items():
        com_set[k] = set(v)
    hyelst = list(hye)
    conutincom = 0
    for e in hyelst:
        for i in com.keys():
            if set(e).issubset(com[i]):
                conutincom += 1
                continue
        pass
    return conutincom


def comlist_to_matrix(h_true_pathcom):
    """
    将真实社区转变为矩阵形式
    h_true_pathcom: 社区路径
    """
    Htruecom = get_true_com(h_true_pathcom)  # 真实社区
    # 将真实社区表示为二维数组
    nodeN = len(set(n for com in Htruecom.values() for n in com))
    comN = len(Htruecom.keys())
    U_true = np.zeros((nodeN, comN))
    for k, v in Htruecom.items():
        for id in v:
            U_true[id][k] = 1
    return U_true


def biology_hit(A: List[Union[list, tuple]], B: List[Union[list, tuple]], omega):
    """
    计算hit(A,B)，公式如下：

    Affinity(A,B) = \frac{(|A \cap B|)^2}{|A| * |B|}

    Hit(A,B) = {A_i \in A| Affinity(A_i,B_i)>\omega, \exists B_j \in B}
    """
    hit_AB = set()
    for a in A:
        for b in B:
            if len(set(a)) * len(set(b)) == 0:
                affinity = 0
            else:
                affinity = (len(set(a) & set(b))) ** 2 / (len(set(a)) * len(set(b)))
                if affinity > omega:
                    hit_AB.add(tuple(a))
    return hit_AB


def biology_f1_score(
        predict_substance: List[Union[list, tuple]], truth_substance: List[Union[list, tuple]], omega: float = 0.2
):
    """
    """
    Recall = len(biology_hit(truth_substance, predict_substance, omega)) / len(truth_substance)
    Precision = len(biology_hit(predict_substance, truth_substance, omega)) / len(predict_substance)

    if (Recall + Precision) > 0:
        f_measure = 2 * Recall * Precision / (Recall + Precision)
    else:
        f_measure = 0
    return f_measure


def partition(u: np.ndarray, threshold: float = 0.3) ->List[List]:
    """
    用于划分重叠社区
    """
    result = []
    for com in range(u.shape[1]):
        node = np.where(u[:,com] > threshold)[0]
        result.append(node.tolist())

    return result

# 创建重叠标签
def create_labels(matrix, threshold):
    labels = []
    for row in matrix:
        # 获取隶属度大于阈值的社区作为标签
        labels.append([i for i, value in enumerate(row) if value > threshold])
    return labels

def calculate_nmi(precom, truecom, threshold: float = 0.3):
    """
    计算重叠社区结构的NMI值

    :param precom: 第一个重叠社区划分（每个节点属于多个社区的情况）
    :param truecom: 第二个重叠社区划分（每个节点属于多个社区的情况）
    :return: NMI值
    """

    prelabel = create_labels(precom, threshold)
    truelabel = create_labels(truecom, threshold)

    nmi = normalized_mutual_info_score(prelabel, truelabel)
    return nmi





if __name__ == "__main__":
    """
    truelabel_0 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    truelabel_1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    # my model
    predlabel_0 = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    predlabel_1 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]

    # comparing model
    predlabel_1_forc = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    predlabel_0_forc = [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]

    num_classes = 2
    
    true_positives, false_positives, false_negatives = calculate_multiclass_metrics(predlabel_1, truelabel_1, num_classes)
    avg_f1_score, f1_scores = calculate_f1_score(true_positives, false_positives, false_negatives)
    print("Average F1 Score for u1:", avg_f1_score)
    print("F1 Scores per Class for u1:", f1_scores)
    
    true_positives, false_positives, false_negatives = calculate_multiclass_metrics(predlabel_0, truelabel_0, num_classes)
    avg_f1_score, f1_scores = calculate_f1_score(true_positives, false_positives, false_negatives)
    print("Average F1 Score for u1:", avg_f1_score)
    print("F1 Scores per Class for u1:", f1_scores)    

    true_positives, false_positives, false_negatives = calculate_multiclass_metrics(predlabel_1_forc, truelabel_1, num_classes)
    avg_f1_score, f1_scores = calculate_f1_score(true_positives, false_positives, false_negatives)
    print("Average F1 Score for u1:", avg_f1_score)
    print("F1 Scores per Class for u1:", f1_scores) 
"""
    # cal_f1(truelabel_0, predlabel_0, num_classes)
    # pre_com_dic = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8, 9]}
    # true_com_dic = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]}
    # s = eval_scores(pre_com_dic[0], true_com_dic[0])
    truth = [(0, 1), (2, 3, 4, 5), (4, 5, 6)]
    predict = [(1, 2, 3, 5), (0, 1, 2, 4, 5), (4, 5, 6)]
    f1 = biology_f1_score(predict, truth, omega=0.9)
    print(f1)
    pass
