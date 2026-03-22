import torch
from micrograd.engine import Value

# 测试思路：以 PyTorch 作为正确答案的参照系，验证 micrograd 的实现是否一致
# 每个测试函数都是同一个套路：
#   1. 用 micrograd 的 Value 做一遍计算，调 backward()
#   2. 用 PyTorch 做完全相同的计算，调 backward()
#   3. 对比两边的 forward 结果（y 的值）和 backward 结果（x 的梯度）
# PyTorch 的 .double() 和 micrograd 的 Value 类似，都是存一个数值并支持自动求导

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True          # 告诉 PyTorch 需要对 x 求梯度
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()      # 对比 forward 结果：两边算出的 y 值一致
    # backward pass went well
    assert xmg.grad == xpt.grad.item()      # 对比 backward 结果：两边算出的 x 的梯度一致

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol        # 对比 forward 结果：两边算出的 g 值一致（允许极小误差）
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol        # 对比 backward 结果：a 的梯度一致
    assert abs(bmg.grad - bpt.grad.item()) < tol        # 对比 backward 结果：b 的梯度一致
