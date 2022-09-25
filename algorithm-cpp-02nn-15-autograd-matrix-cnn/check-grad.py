import numpy as np
import torch
import torch.nn

def dtype2int(dtype : np.dtype):
    
    if dtype == np.float32:
        return 0
    elif dtype == np.float16:
        return 1
    elif dtype == np.float64:
        return 2
    elif dtype == np.int32:
        return 3
    elif dtype == np.uint32:
        return 4
    elif dtype == np.int64:
        return 5

    assert False, f"Unsupport dtype {dtype}"


def int2dtype(itype : int):
    
    if itype == 0:
        return np.float32
    elif itype == 1:
        return np.float16
    elif itype == 2:
        return np.float64
    elif itype == 3:
        return np.int32
    elif itype == 4:
        return np.uint32
    elif itype == 5:
        return np.int64

    assert False, f"Unsupport itype {itype}"


def load_tensor(file):
            
    with open(file, "rb") as f:
        binary_data = f.read()
        
    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
    
    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)
    np_dtype = int2dtype(dtype)
    return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)


def save_tensor(file, tensor : np.ndarray):
            
    with open(file, "wb") as f:
        f.write(np.array([0xFCCFE2E2, tensor.ndim, dtype2int(tensor.dtype)] + list(tensor.shape), dtype=np.uint32).tobytes())
        f.write(tensor.tobytes())


def load_tt(file):
    t = load_tensor(file)
    return torch.tensor(t)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 5, 3, 1, 0)
        self.conv1.weight.data[:] = load_tt("workspace/param0.bin")
        self.conv1.bias.data[:] = load_tt("workspace/param1.bin")
        self.fc1 = torch.nn.Linear(3380, 10)
        self.fc1.weight.data[:] = load_tt("workspace/param2.bin").T
        self.fc1.bias.data[:] = load_tt("workspace/param3.bin")

    def forward(self, x):
        x = torch.relu(self.conv1(x)).view(-1, 3380)
        return self.fc1(x)

# 这个程序用来检验c++的求导过程是否正确，需要配合c++中的参数导出部分

model = Model()
input = load_tensor("workspace/input.bin")
output = load_tensor("workspace/output.bin")
input_p = torch.nn.parameter.Parameter(torch.tensor(input))
output_torch = model(input_p)
output_torch.sum().backward()
print(np.abs(output_torch.detach().numpy() - output).sum(), "=======outputgrad")

input_grad = load_tensor("workspace/input.grad.bin")
print(np.abs(input_p.grad.data - input_grad).sum(), "=======input")

p0_grad = load_tensor("workspace/param0.grad.bin")
print(np.abs(model.conv1.weight.grad.data - p0_grad).max(), "=======conv1.weight.grad")

p1_grad = load_tensor("workspace/param1.grad.bin")
print(np.abs(model.conv1.bias.grad.data - p1_grad).sum())

p2_grad = load_tensor("workspace/param2.grad.bin")
print(np.abs(model.fc1.weight.grad.data - p2_grad.T).sum())

p3_grad = load_tensor("workspace/param3.grad.bin")
print(np.abs(model.fc1.bias.grad.data - p3_grad).sum())
