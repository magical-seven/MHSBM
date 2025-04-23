from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Union, List

import numpy as np

class MultiHypergraph(ABC):
    N: int

    @abstractmethod
    def get_layername_list(self) ->Any:
        """返回层列表"""

    @ abstractmethod
    def get_layerNum(self):
        """
        获取层的数量
        """

    @abstractmethod
    def get_node_num(self):
        """
        Returns
        -------
        获取每一层的节点数目
        """

    @abstractmethod
    def get_D_C(self, D_C: dict):
        """
        整理度矩阵与常数值
        """
    @abstractmethod
    def creat_LayerLink_matrix(self):
        """
        创建层与层的连接矩阵
            若层l1与l2存在层间连接，则layerlink_matrix[l1,l2]=1
        """

    @abstractmethod
    def get_max_hye_dic(self, DLbool: bool = True) -> Union[Dict, List]:
        '''
        description:获取超边列表或字典，层名作为索引

        param {bool} DLbool

        return {*}
        '''


    @abstractmethod
    def get_BA(self):
        '''
            Returns

            -------

            获取B*A的矩阵

        '''
