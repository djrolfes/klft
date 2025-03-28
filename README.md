# klft
A library for lattice field theory simulation accelerated using Kokkos

## Installation
```bash
cmake -DKokkos_ROOT=/path/to/kokkos/installation [-DCMAKE_CXX_COMPILER=/path/to/kokkos/nvcc/wrapper] /path/to/klft
make
```

## Usage

### Metropolis
```bash
Usage: ./metropolis [options]
Options:
--gauge-group SU2 or U1
--ndim 2, 3, or 4
--LX lattice size in x direction
--LY lattice size in y direction
--LZ lattice size in z direction
--LT lattice size in t direction
--n-hit number of hits per sweep
--beta inverse coupling constant
--delta step size
--seed random number generator seed
--n-sweep number of sweeps
--cold-start true or false
--outfilename output filename
```

### HMC
```bash
Usage: ./hmc [options]
Options:
--gauge-group SU2 or U1
--ndim 2, 3, or 4
--LX lattice size in x direction
--LY lattice size in y direction
--LZ lattice size in z direction
--LT lattice size in t direction
--n-traj number of trajectories
--n-steps number of steps in leapfrog
--tau trajectory length
--beta inverse coupling constant
--seed random number generator seed
--outfilename output filename
```