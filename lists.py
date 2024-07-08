import itertools
from collections.abc import Callable
from typing import *

from bitarray import *
import random


# "configurations" here are just bitmask representations of the truth of variables. e.g. bit 0 could correspond to "A".

class TruthTable:
    def __init__(self, nVars: int, table):
        self.n = nVars
        if table is None:
            self.table = bitarray(1 << nVars)
        else:
            self.table = table
            assert len(table) == (1 << nVars)

    def populateTable(self, percentTruth: float):
        r = random
        for i in range(len(self.table)):
            if r.random() < percentTruth:  # populates each configuration with a percentTruth chance of being true
                self.table[i] = 1

    # or and and are basic operations that allow for a quick calculation of configurations with duplicates by taking the
    # disjunction of pairwise ands for identical values

    def __or__(self, other):
        assert other.n == self.n
        return TruthTable(self.n, other.table | self.table)

    def __and__(self, other):
        assert other.n == self.n
        return TruthTable(self.n, other.table & self.table)

    def __getitem__(self, item):
        return self.table[item]

    def __len__(self):
        return 1 << self.n


# really just a tuple with the ability to access the truth of the item given a configuration

class VariationalElement:
    def __init__(self, val: int, variation: TruthTable):
        self.val = val
        self.var = variation

    def __getitem__(self, ind):
        return self.var[ind]

    def __repr__(self):
        return str(self.val) + ", " + str(self.var.table)


class VariationalList:
    def __init__(self, lst: List[VariationalElement]):
        self.lst = lst

    def __getitem__(self, ind):
        return tuple([i.val for i in self.lst if i[ind]])

    def __repr__(self):
        return repr(self.lst)


class FeatureModel:
    def __init__(self, table: TruthTable):
        self.table = table
        self.configurations = []
        for i in range(len(self.table)):
            if table.table[i] == 1:
                self.configurations.append(i)

    def applyModelToList(self, lst: VariationalList):
        data = {}
        for configuration in self.configurations:
            product = lst[configuration]
            if product not in data:
                data[product] = []
            data[product].append(configuration)
        return ProductLine(self, data, lst)

    def __repr__(self):
        return repr(self.table)


class ProductLine:
    def __init__(self, fm: FeatureModel, data: Dict[Tuple[int, ...], List[int]], lst: VariationalList):
        self.fm = fm
        self.data = data
        self.lst = lst

    def __len__(self):
        return len(self.data)

    def filterNoDuplicates(self):
        nodup = {}
        for i in self.data:
            if len(set(i)) == len(i):  # very naive way of finding duplicates but this runs in O(len*|fm|)
                nodup[i] = self.data[i]
        return ProductLine(self.fm, nodup, self.lst)

    @staticmethod
    def isSorted(lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def filterSorted(self):
        sorted = {}
        for i in self.data:
            if ProductLine.isSorted(i):  # very naive way of finding duplicates but this runs in O(len*|fm|)
                sorted[i] = self.data[i]
        return ProductLine(self.fm, sorted, self.lst)

    def __repr__(self):
        return "Product Line over " + repr(self.lst) + " and data " + repr(self.data)


def generateVariationalList(listSize: int, listDistribution: Callable[[], int], nVars: int,
                            percentTruthDistribution: Callable[[], float]):
    lst = []
    for i in range(listSize):
        # make the necessary calls to the random distributions, and generate a list.
        table = TruthTable(nVars, None)
        table.populateTable(percentTruthDistribution())
        lst.append(VariationalElement(listDistribution(), table))
    return VariationalList(lst)





if __name__ == "__main__":
    testCase = generateVariationalList(8, lambda: random.randint(1, 5), 2, lambda: 0.3)
    print(testCase)
    fmTruth = TruthTable(2, None)
    fmTruth.populateTable(1)
    fm = FeatureModel(fmTruth)
    productLine = fm.applyModelToList(testCase)
    print(productLine)
    print(productLine.filterNoDuplicates())
    print(productLine.filterSorted())
