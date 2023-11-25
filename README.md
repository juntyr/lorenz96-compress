# Exploring online compression the Lorenz96 model

This project was developed during the ESiWACE3 hackathon at CSC Finland on 18.10. - 20.10.2023 by Victor Azizi from the Netherlands eScience Center and Juniper Tyree from the University of Helsinki.

## Compiling ZFP for LUMI-G

This sections details the process to compile ZFP on LUMI-G. Note that the Makefile should take care of the entire process - the following are Victor Azizi's notes on how he got ZFP to compile:

- First, check out the zfp submodule:
  ```bash
  git submodule init
  git submodule update
  ```
  Alternatively, clone the repository and check out the staging branch (checked with hash e935a81):
  ```bash
  git clone https://github.com/LLNL/zfp.git
  cd zfp
  git checkout staging
  ```
- Next, load the right modules on LUMI:
  ```bash
  module load LUMI
  module load buildtools
  ```
  They are also available in the `modules` file, which can be `source`d.
- Now, configure the cmake build of ZFP:
  ```bash
  mkdir -p build
  cd build
  cmake -DZFP_WITH_HIP=ON -DZFP_WITH_OPENMP=OFF -DHIP_PATH=/opt/rocm/hip -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DBUILD_TESTING=OFF ../
  ```
- Next, build with VERBOSE, as you will manually have to fix some commands
  ```bash
  make VERBOSE=1
  ```
- You now have a library, but the compilation step of `zfp.dir/hip/interface.cpp.o` shows that the wrong offload target was used. Therefore, you have to manually fix the offload target to `gfx90a`:
  ```bash
  cd build
  CC -DZFP_ROUNDING_MODE=ZFP_ROUND_NEVER -DZFP_WITH_HIP -Dzfp_EXPORTS -I../../include -O3 -DNDEBUG --offload-arch=gfx90a -fPIC -std=gnu++14 -o CMakeFiles/zfp.dir/hip/interface.cpp.o -x hip -c ../../src/hip/interface.cpp
  ```
- Finally, rebuild the library:
  ```bash
  cd build
  make
  ```  
- Success! The AMD GPU enabled library can now be found in `zfp/build/lib64/libzfp.so`

## Reproducing the plot notebooks from the command line

- First, clear the previous output:
  ```bash
  jupyter nbconvert --clear-output --inplace plot.ipynb
  ```
- Second, use `nbconvert` or `papermill` to execute the notebook, e.g.:
  ```bash
  papermill --autosave-cell-every 5 --progress-bar plot.ipynb plot.nbconvert.ipynb
  ```
- Optionally, remove execution metadata:
  ```bash
  jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --to=notebook --inplace plot.nbconvert.ipynb
  ```
- Finally, replace the old notebook with the new:
  ```bash
  mv plot.nbconvert.ipynb plot.ipynb
  ```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Citation

Please cite the authors of this project and this repository.

## Funding

This project has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
