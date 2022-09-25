import numpy as np

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


tensor = np.arange(50).reshape(5, 2, 5)
save_tensor("workspace/test.tensor", tensor)

loaded = load_tensor("workspace/test.tensor")
print(loaded, loaded.shape, loaded.dtype)

print(np.allclose(tensor, loaded))