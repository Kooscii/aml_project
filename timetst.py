from timeit import timeit

# s = """\
# a = [[i for i in range(1000000)],[i+1000000 for i in range(1000000)]]
# b = [i for i in range(2000000)]
# d = set(b)-set(a[0])-set(a[1])
# """

s = """\
a = [i for i in range(20000)]
b = [i*2+1 for i in range(10000)]
c = [i*2 for i in range(10000)]
d = [[],[]]
d[0] = list(set(a)-set(b))
d[1] = list(set(a)-set(d[0]))
print(d)
"""

timeit(stmt=s, number=10)
