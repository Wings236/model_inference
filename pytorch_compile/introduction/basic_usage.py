import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

t1 = torch.randn(10, 10)
t2 = torch.randn(10, 10)

# 直接调用得到函数
opt_foo1 = torch.compile(foo)
print(opt_foo1(t1, t2))

# 修饰函数
@torch.compile
def foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

print(foo2(t1, t2))


t = torch.randn(10, 100)
class MyMoudle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyMoudle()
mod.compile()
print(mod(t))

