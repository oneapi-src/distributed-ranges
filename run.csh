#[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device 1.2 [2022.15.12.0.01_081451]
#[opencl:cpu:1] Intel(R) OpenCL, Intel(R) Xeon(R) Platinum 8480+ 3.0 [2022.14.10.0.20_160000.xmain-hotfix]
#[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Graphics [0x0bd5] 1.3 [1.3.24595]
#[ext_oneapi_level_zero:gpu:1] Intel(R) Level-Zero, Intel(R) Graphics [0x0bd5] 1.3 [1.3.24595]
#[ext_oneapi_level_zero:gpu:2] Intel(R) Level-Zero, Intel(R) Graphics [0x0bd5] 1.3 [1.3.24595]
#[ext_oneapi_level_zero:gpu:3] Intel(R) Level-Zero, Intel(R) Graphics [0x0bd5] 1.3 [1.3.24595]


#srun -p QZ1J-ICX-PVC ./build/examples/shp/dot_product_benchmark        # 1GPU
#srun -p QZ1J-SPR-PVC-2C ./build/examples/shp/dot_product_benchmark     # 2GPUs
SIZE=8192
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 1 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 2 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 3 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 4 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 5 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 6 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 7 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 8 $SIZE

SIZE=12243
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 1 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 2 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 3 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 4 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 5 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 6 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 7 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 8 $SIZE

SIZE=16384
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 1 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 2 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 3 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 4 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 5 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 6 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 7 $SIZE
srun -p QZ1B-SPR-4oam-PVC ./build/examples/shp/dot_product_benchmark 8 $SIZE
