# 下载依赖
- `kiwi install kiwi-requirement.txt`
    - 通过这个指令下载依赖到当前目录

# 如果静态库不能使用
- 请通过`kiwi get-data OpenBLAS-0.3.17-Source`
    - 下载后，`cd OpenBLAS-0.3.17-Source`
    - 然后进行编译，`bash openblas_build_and_install.sh`