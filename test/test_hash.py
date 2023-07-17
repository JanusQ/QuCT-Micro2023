

class A():
    def __hash__(self):
        return hash(1)

class B():
    def __hash__(self):
        return hash(1)

# d = {}
# d[A()] = 1
# print(d[B()])

s = set()
s.add(A())
s.add(A())
print(len(s))