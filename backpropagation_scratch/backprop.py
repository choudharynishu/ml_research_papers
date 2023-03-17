'''
This is a step-by-step implementation of gradient descent, inspired by Dr. Karpathy's micrograd
Nishu Choudhary
'''
import math
#Value object for derivative operations
class value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"value(data= {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data +other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, value) else value(other)
        out = value(self.data*other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int,float)), "Only integer and floating points for powers"
        out = value(self.data**power, (self,), f'**{power}')
        def _backward():
            self.grad += power*self.data**(power-1)*out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        out = value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1-t**2)* out.grad
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        out = value(max(0.0, x), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0)*out.grad
        out._backward = _backward
        return out

    #Topological sort
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()




