
# 神经网络训练的三个阶段：
#
# 1. Forward（前向传播）：用当前参数计算预测值，再算出 loss（预测有多错）
#       y_pred = w1*x1 + w2*x2
#       loss   = (y_pred - y_true) ** 2
#
# 2. Backward（反向传播）：对 loss 求导，算出每个参数的梯度
#       loss.backward()
#       # 执行完后 w1.grad = dLoss/dw1，w2.grad = dLoss/dw2
#       # 梯度的含义：该参数增大一点，loss 会增大还是减小，幅度多大
#
# 3. 更新参数 / Optimization（优化）：沿梯度反方向，走一小步（学习率 lr 控制步子大小）
#       w1.data -= lr * w1.grad
#       w2.data -= lr * w2.grad
#    具体的更新算法叫 Optimizer（优化器），最基础的叫 SGD（Stochastic Gradient Descent，随机梯度下降）
#    PyTorch 中：
#       optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#       optimizer.step()       # 执行参数更新
#       optimizer.zero_grad()  # 梯度清零，准备下一轮
#
# 反复执行以上三步，loss 越来越小，模型越来越准。
#
# PyTorch 封装程度说明：
#   - loss.backward()：PyTorch 自动完成第 2 步（求导），这正是 micrograd 手动实现的部分
#   - Loss 函数和 Optimizer：需要自己选，PyTorch 提供了很多现成选项
#   - 用现成网络结构（Linear、CNN 等）：连第 1 步 forward 计算也封装好了
#   所以用 PyTorch 训练模型，自己写的核心只有三行：
#       loss = criterion(y_pred, y_true)  # 选 Loss 函数，算出 loss
#       loss.backward()                   # 自动求导
#       optimizer.step()                  # 更新参数
#
# Value 类的作用：在 forward 时自动记录计算图，使得 backward 可以自动完成第 2 步。

# ─────────────────────────────────────────────────────────────────────────────
# Loss 函数：衡量"模型预测有多错"的函数，训练目标就是让 loss 尽量小。
# 不同任务选不同的 loss 函数，以下是三种最常见的：
#
# 1. MSE，均方误差（Mean Squared Error）
#       loss = (y_pred - y_true) ** 2        # 单样本
#       loss = sum((y_pred - y_true) ** 2) / n  # 多样本取平均
#    适用：回归问题，预测连续值，比如预测房价、温度
#    特点：误差越大惩罚越重（平方会放大大误差），对离群点敏感
#
# 2. MAE，平均绝对误差（Mean Absolute Error）
#       loss = abs(y_pred - y_true)          # 单样本
#       loss = sum(abs(y_pred - y_true)) / n # 多样本取平均
#    适用：回归问题，但数据中存在较多离群点时比 MSE 更稳健
#    特点：所有误差同等对待，不放大大误差
#
# 3. 交叉熵（Cross-Entropy Loss）
#       loss = -log(y_pred)                                               # 直觉公式：只看正确类别的预测概率
#       loss = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))  # 二分类完整公式
#    适用：分类问题，比如判断图片是猫还是狗、垃圾邮件分类
#    特点：y_pred 是模型输出的概率（0~1），对"自信地答错"惩罚极重：
#           预测概率越接近 0，loss 爆炸式增长（-log(0.01) = 4.6），
#           迫使模型不敢对错误答案表现出高置信度
# ─────────────────────────────────────────────────────────────────────────────

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data                    # 这个节点存的标量值，比如 3.0
        self.grad = 0                       # 对最终输出 L 的梯度 dL/d(self)，初始为 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None       # 反向传播函数，由各运算符在 forward 时定义；叶子节点默认什么都不做
        self._prev = set(_children)         # 产生这个节点的输入节点集合，用于构建计算图
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 允许和普通数字相加，自动包装成 Value
        out = Value(self.data + other.data, (self, other), '+')      # forward：标量相加，记录父节点和运算符

        def _backward():
            # out = self + other，对 self 和 other 的偏导都是 1
            # 链式法则：dL/d(self) += dL/d(out) * d(out)/d(self) = out.grad * 1
            self.grad += out.grad           # += 而非 =：一个节点可能被多条路径使用，梯度需要累加
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 允许和普通数字相乘
        out = Value(self.data * other.data, (self, other), '*')      # forward：标量相乘

        def _backward():
            # out = self * other，对 self 的偏导是 other.data，对 other 的偏导是 self.data
            # 链式法则：dL/d(self) += dL/d(out) * d(out)/d(self) = out.grad * other.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')  # forward：幂运算；指数是普通数字，不是 Value，不参与计算图

        def _backward():
            # out = self^n，导数为 n * self^(n-1)
            # 链式法则：dL/d(self) += dL/d(out) * n * self.data^(n-1)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')  # forward：ReLU(x) = max(0, x)

        def _backward():
            # ReLU 导数：x > 0 时为 1，x <= 0 时为 0
            # (out.data > 0) 返回 Python bool，乘以 out.grad 实现分段导数
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        # 拓扑排序：DFS 后序遍历，确保每个节点排在它所有子节点之后
        # 最终 topo 顺序：叶子节点在前，输出节点（self）在最后
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)       # 先递归处理所有子节点
                topo.append(v)              # 子节点都处理完后，自己才入队
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # reversed(topo)：从输出节点往叶子方向逆序遍历，保证调用 _backward 时 out.grad 已经就绪
        self.grad = 1                       # dL/dL = 1，Loss 对自身的梯度是 1，反向传播的起点
        for v in reversed(topo):
            v._backward()                   # 每个节点把梯度传给自己的子节点，链式法则逐层传播

    def __neg__(self): # -self
        return self * -1                    # 复用 __mul__，无需新定义 _backward

    def __radd__(self, other): # other + self
        return self + other                 # 处理 3 + value 这种左操作数是普通数字的情况

    def __sub__(self, other): # self - other
        return self + (-other)              # 减法 = 加法 + 取负，复用已有操作

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other                 # 处理 3 * value 这种左操作数是普通数字的情况

    def __truediv__(self, other): # self / other
        return self * other**-1             # 除法 = 乘以倒数，复用 __mul__ 和 __pow__，梯度自动正确

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
