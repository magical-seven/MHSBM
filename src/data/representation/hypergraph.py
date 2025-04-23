from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np


class Hypergraph(ABC):  # 定义一个抽象类
    """Abstract class for the representation of hypergraphs."""

    N: int  # Number of nodes.
    E: int  # Number of hyperedges.
    max_hye_size: int  # Maximum size of the hyperedges in the hypergraph.
    hye_count: Dict[
        int, int
    ]  # Hyperedges divided by hyperedge size, as (key, value) pairs: (size, count).

    @abstractmethod
    def get_repr(self) -> Any:
        """Return the internal representation of the hypergraph.
        返回超图的内部表示形式
        """

    @abstractmethod
    def get_incidence_matrix(self) -> Any:
        """Return the incidence matrix B.
        返回关联矩阵B，带权重
        """

    @abstractmethod
    def get_binary_incidence_matrix(self) -> Any:
        """Return the incidence matrix B with only zeros and ones.
        返回只有0和1的关联矩阵  无向图
        """

    @abstractmethod
    def get_hye_weights(self) -> Any:
        """Return the edge weight array.
        返回边权重数组
        """

    @abstractmethod
    def __iter__(self) -> Iterable[Any, Union[int, float]]:
        """Create an iterable that yields (hyperedge, weight) tuples.
        创建一个可迭代的元组(超边，权重)
        """


    def sub_hyg(self, *args: Any) -> Hypergraph:
        """Return a sub-hypergraph representation."""
        raise NotImplementedError(f"Not implemented for instance of {self.__class__}")

    def save_to_txt(self, savedir: Union[str, Path]) -> None:
        savedir = Path(savedir)

        with open(savedir / "hyperedges.txt", "w") as hye_file:
            for hye, _ in self:
                hye_file.write(" ".join(map(str, hye)) + "\n")

        np.savetxt(
            savedir / "weights.txt",
            self.get_hye_weights().astype(int),
            fmt="%i",
        )

    def load(self, *args, **kwargs) -> Any:
        """Load the hypergraph from external sources."""
        raise NotImplementedError(f"Not implemented for instance of {self.__class__}")
