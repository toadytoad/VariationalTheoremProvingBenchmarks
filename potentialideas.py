# This file is just a collection of methods that were scrapped during code cleanup. Things that might become useful
# later but not relevant at the moment.
import itertools

from bitarray import bitarray

from lists import TruthTable
from sympy import *
from string import ascii_lowercase


def serializeTestCase(lst, file):
    with open(file, "wb") as f:
        f.write(b'ndup')
        f.write(len(lst).to_bytes(4, 'big'))
        f.write(lst[0].var.n.to_bytes(4, 'big'))
        for i in lst:
            f.write(i.val.to_bytes(4, 'big'))
            f.write(i.var.table.tobytes())
        f.close()


def serializeTestCaseSolution(sol: bitarray, file):
    with open(file, "wb") as f:
        f.write(b'\xeedup')  # same string as test case just with the msb of "n" set to 1
        f.write(sol.tobytes())
        f.close()


def solveNoDupWithTruthTable(lst):
    buckets = {}
    for i in lst:  # sort the items into buckets where they may be duplicates
        if i.val not in buckets:
            buckets[i.val] = []
        buckets[i.val].append(i.var)
    ba = bitarray(len(lst[0].var.table))
    for b in buckets:
        bucket = buckets[b]
        for i, j in itertools.combinations(bucket, 2):
            ba |= i.table & j.table  # disjunction of pairwise ands to get truth satisfiability of configuration
        # Note: If you want to create a test case that is satisfiable for all configurations, add some code here that
        # goes through i.table&j.table and for any 1's it finds nuke the 1 in i or j (or both) so that the and of the
        # two now give a 0. This takes a little refactoring but is not too hard, and creates a variational problem which
        # is provable for all variations
    return ba


# For creating boolean expressions out of truth tables:
# Define several z3 boolean references which are enumerated and properly assigned to each bit of the truth table index.
# Using the variables create a z3 CNF and apply z3's simplify function on it.
# Ok further research and testing shows z3 is terrible for this
# Instead use sympy
# e.g
#
# a, b, c =symbols('a b c')
# expr = (~a&~b&~c)|(~a&b&c)|(a&~b&c)|(a&b&~c)|(a&b&c)
# simplify_logic(expr)
# (a & b) | (a & c) | (b & c) | (~a & ~b & ~c)

def truthTableToBooleanExpression(table: TruthTable):
    n = table.n
    s = symbols(' '.join(list(ascii_lowercase[:n])))
    expr = false
    for i in range(1 << n):
        if table[i] == 0:
            continue

        subExpr = True
        for j in range(n):
            if i & 1:
                subExpr &= s[j]
            else:
                subExpr &= ~s[j]
            i >>= 1
        expr |= subExpr
    print(expr)
    return simplify_logic(expr)


table = TruthTable(6, None)
table.populateTable(0.8)
print(truthTableToBooleanExpression(table))
