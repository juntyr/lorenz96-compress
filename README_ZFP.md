## How to compile ZFP

- check out the zfp submodule (or skip to the next step)
  ```bash
  git submodule init
  git submodule update
- clone the repository and check out the staging branch (checked with hash e935a) (if the previous step failed):
  ```bash
  git clone https://github.com/LLNL/zfp.git
  cd zfp
  git checkout staging
- load the right modules (put them in a modulefile to speed it up):
  ```
  module load LUMI
  module load buildtools
- do the cmake magic
  ```
  mkdir build
  cd build
  cmake -DZFP_WITH_HIP=ON -DZFP_WITH_OPENMP=OFF -DHIP_PATH=/opt/rocm/hip -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DBUILD_TESTING=OFF ../
  
- build, but VERBOSE because we will manually have to fix some commands
  ```
  make VERBOSE=1
- great! we now have a binary, but the compilation step of `zfp.dir/hip/interface.cpp.o` will show you that the offload target is wrong! :anguished: 
- Manually "fix" the offload target to gfx90a (dont forget to change to your username)
  ```
   cd /users/vazizi2/zfp/build/src && /opt/rocm-5.2.3/llvm/bin/clang++ -DZFP_ROUNDING_MODE=ZFP_ROUND_NEVER -DZFP_WITH_HIP -Dzfp_EXPORTS -I/users/vazizi2/zfp/include -O3 -DNDEBUG --offload-arch=gfx90a -fPIC -std=gnu++14 -o CMakeFiles/zfp.dir/hip/interface.cpp.o -x hip -c /users/vazizi2/zfp/src/hip/interface.cpp
- Remake our library, return to `zfp/build/` first
  ```
  cd ~/zfp/build
  make
- Success! AMD GPU enabled binary can now be found in `zfp/build/bin/zfp
