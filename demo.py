import random
import numpy as np
import matplotlib.pyplot as plt
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
from sklearn.datasets import make_moons

# 固定随机种子，保证每次运行结果一致
np.random.seed(1337)
random.seed(1337)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 生成数据集
# ─────────────────────────────────────────────────────────────────────────────

X, y = make_moons(n_samples=100, noise=0.1)
# X：shape (100, 2)，每个样本是平面上一个点的坐标 (x1, x2)
# y：shape (100,)，标签，原始值是 0 或 1

y = y * 2 - 1
# 把标签从 {0, 1} 转换成 {-1, 1}，因为后面的 SVM loss 需要这种格式

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')
# 可视化数据集：两条交叉的月牙形，一条红一条蓝，训练目标是让模型学会分开它们
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 2. 初始化模型
# ─────────────────────────────────────────────────────────────────────────────

model = MLP(2, [16, 16, 1])
# 输入 2 个特征，两个隐藏层各 16 个神经元，输出 1 个值（正负代表两个类别）
# 参数总量 337，计算方式：
#   Layer(2→16)：  16 × (2个权重 + 1个偏置) = 16 × 3  =  48
#   Layer(16→16)： 16 × (16个权重 + 1个偏置) = 16 × 17 = 272
#   Layer(16→1)：   1 × (16个权重 + 1个偏置) =  1 × 17 =  17
#   总计：48 + 272 + 17 = 337
print(model)
print("number of parameters", len(model.parameters()))

# ─────────────────────────────────────────────────────────────────────────────
# 3. 定义 Loss 函数
# ─────────────────────────────────────────────────────────────────────────────

def loss(batch_size=None):

    # 取数据：支持 mini-batch（随机抽一小批），不传则用全部数据
    # mini-batch 是 SGD 里 "Stochastic（随机）" 的含义，不每次用全部样本，速度更快
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]    # 随机打乱后取前 batch_size 个索引
        Xb, yb = X[ri], y[ri]

    # 把 numpy 数组的每一行转成 Value 列表，才能进入计算图参与 backward
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward：把每个输入样本喂给模型，得到预测分数
    scores = list(map(model, inputs))

    # SVM Hinge Loss：max(0, 1 - y * score)，用 relu 实现
    # y 是正确标签（-1 或 1），score 是模型输出
    # 预测正确且置信度超过边界（y * score >= 1）→ loss = 0，不惩罚
    # 预测错误或置信度不足 → loss > 0，惩罚
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))  # 对所有样本取平均

    # L2 正则化：惩罚过大的权重，防止过拟合，让模型保持简单
    # 把所有参数的平方加起来，乘一个很小的系数 alpha 加到 loss 里
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))

    total_loss = data_loss + reg_loss
    # total_loss 同时优化两个目标：预测要准（data_loss），参数要小（reg_loss）

    # 顺便计算准确率：预测符号和标签符号一致则正确
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()
print(total_loss, acc)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 训练循环：forward → backward → 更新参数，反复 100 次
# ─────────────────────────────────────────────────────────────────────────────

for k in range(100):

    # forward：用当前参数算出 loss
    total_loss, acc = loss()

    # backward：梯度清零后，反向传播算出所有参数的梯度
    model.zero_grad()           # 必须先清零，否则梯度会累加到上一轮
    total_loss.backward()

    # 更新参数（SGD）：沿梯度反方向走一步
    # 学习率从 1.0 线性衰减到 0.1，训练初期步子大，后期步子小，有助于稳定收敛
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 可视化决策边界
# ─────────────────────────────────────────────────────────────────────────────

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# 在整个平面上密集撒点，让模型对每个点做预测，从而画出决策边界
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])     # 预测为正类（score > 0）的区域涂一种颜色
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)   # 背景色表示模型的决策区域
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)  # 原始数据点叠加在上面
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
