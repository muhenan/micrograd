import random
from micrograd.engine import Value

# nn.py 在 engine.py 的 Value 基础上搭积木，构建神经网络
# 三层嵌套结构，从小到大：
#   Neuron（单个神经元）← Layer（一层）← MLP（整个网络）

class Module:
    # 所有网络组件的基类，提供两个公共方法

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0          # 每轮训练结束后必须调，否则梯度会累加到下一轮

    def parameters(self):
        return []               # 占位符，子类各自覆盖，返回自己真正的参数列表

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # 权重列表，每个输入对应一个，随机初始化在 [-1, 1]
        self.b = Value(0)       # 偏置，初始为 0
        self.nonlin = nonlin    # 是否使用 relu 激活，最后一层通常设为 False

    def __call__(self, x):
        # forward：计算 w1*x1 + w2*x2 + ... + b，这是一个神经元的核心计算
        # zip(self.w, x) 把权重和输入配对，逐个相乘后求和，sum 的第二个参数 self.b 是起始值
        # 这样 b 也是 Value 类型，能参与计算图，backward 时会自动求导
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act   # 加 relu 引入非线性，最后一层不加直接输出原值

    def parameters(self):
        return self.w + [self.b]    # 返回所有需要训练的参数：权重 + 偏置，供 optimizer 遍历更新

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        # nout 个神经元，每个神经元接收 nin 个输入
        # 例：Layer(3, 4) 表示 4 个神经元、每个接收 3 个输入
        # **kwargs 透传给每个 Neuron，比如 nonlin=False
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]              # 把同一个输入 x 喂给这层所有神经元
        return out[0] if len(out) == 1 else out         # 只有 1 个神经元时直接返回值，不包在列表里（方便最后一层输出单个数）

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]    # 把这层所有神经元的参数展平成一个列表

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    # MLP（Multi-Layer Perceptron，多层感知机）：最基础的全连接神经网络

    def __init__(self, nin, nouts):
        # nin：输入特征数量
        # nouts：列表，每个元素是对应层的神经元数量
        # 例：MLP(2, [4, 4, 1]) 表示输入 2 个特征，两个隐藏层各 4 个神经元，输出 1 个值
        sz = [nin] + nouts      # sz = size，每一层的大小（神经元数量）
        # 把输入维度拼进去，得到完整的尺寸列表，相邻两个数就是一层的输入→输出大小
        # 例：MLP(2, [4, 4, 1]) → sz = [2, 4, 4, 1]
        #   Layer(2→4)、Layer(4→4)、Layer(4→1)
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        # nonlin=i!=len(nouts)-1：除最后一层外都加 relu，最后一层输出原始值

    def __call__(self, x):
        # forward：把输入依次穿过每一层，上一层的输出是下一层的输入
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]     # 收集所有层的所有参数，一次遍历更新整个网络

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
