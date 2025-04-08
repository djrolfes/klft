//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/

// A very rudimentary tuning setup for tiling
// in Kokkos MDRangePolicy
// This is a work in progress, currently only works well
// for 4D fields and really does some kind of tuning only
// in cudaspace

#pragma once
#include "GLOBAL.hpp"

// define how many times the kernel is run to tune
#ifndef STREAM_NTIMES
#define STREAM_NTIMES 20
#endif

namespace klft
{

  // need to hash the functor to get a unique id
  // to store tuned tiling so that tuning is not repeated
  // over multiple kernel calls
  template <class FunctorType>
  size_t get_Functor_hash(const FunctorType &functor) {
    // this is a very naive hash function
    // should be replaced with a better one
    return std::hash<const void*>{}(&functor);
  }

  // define a hash table to look up the tuned tiling
  // this is a very naive hash table
  // should be replaced with a better one
  template <size_t rank>
  struct TuningHashTable {
    std::unordered_map<size_t, IndexArray<rank>> table;
    void insert(const size_t key, const IndexArray<rank> &value) {
      table[key] = value;
    }
    IndexArray<rank> get(const size_t key) {
      return table[key];
    }
    bool contains(const size_t key) {
      return table.find(key) != table.end();
    }
    void clear() {
      table.clear();
    }
  };

  // create global 4D, 3D and 2D hash tables
  TuningHashTable<4> tuning_hash_table_4D;
  TuningHashTable<3> tuning_hash_table_3D;
  TuningHashTable<2> tuning_hash_table_2D;

  template<size_t rank, class FunctorType>
  void tune_and_launch_for(const IndexArray<rank> &start,
                  const IndexArray<rank> &end,
                  const FunctorType &functor) {
    // launch kernel if tuning is disabled
    if (!KLFT_TUNING) {
      const auto policy = Policy<rank>(start, end);
      Kokkos::parallel_for(policy, functor);
      return;
    }
    // check if the functor is already tuned
    const size_t functor_hash = get_Functor_hash(functor);
    switch(rank) {
      case 4:
        if(tuning_hash_table_4D.contains(functor_hash)) {
          auto tiling = tuning_hash_table_4D.get(functor_hash);
          if (KLFT_VERBOSITY > 2) {
            printf("Tuning found for functor %zu, tiling: %d %d %d %d\n", functor_hash, tiling[0], tiling[1], tiling[2], tiling[3]);
          }
          auto policy = Policy<rank>(start, end, tiling);
          Kokkos::parallel_for(policy, functor);
          return;
        }
        break;
      case 3:
        if(tuning_hash_table_3D.contains(functor_hash)) {
          auto tiling = tuning_hash_table_3D.get(functor_hash);
          if (KLFT_VERBOSITY > 2) {
            printf("Tuning found for functor %zu, tiling: %d %d %d %d\n", functor_hash, tiling[0], tiling[1], tiling[2], tiling[3]);
          }
          auto policy = Policy<rank>(start, end, tiling);
          Kokkos::parallel_for(policy, functor);
          return;
        }
        break;
      case 2:
        if(tuning_hash_table_2D.contains(functor_hash)) {
          auto tiling = tuning_hash_table_2D.get(functor_hash);
          if (KLFT_VERBOSITY > 2) {
            printf("Tuning found for functor %zu, tiling: %d %d %d %d\n", functor_hash, tiling[0], tiling[1], tiling[2], tiling[3]);
          }
          auto policy = Policy<rank>(start, end, tiling);
          Kokkos::parallel_for(policy, functor);
          return;
        }
        break;
      default:
        break;
    }
    // if not tuned, tune the functor
    const auto policy = Policy<rank>(start, end);
    IndexArray<rank> best_tiling;
    for(index_t i = 0; i < rank; i++) {
      best_tiling[i] = 1;
    }
    // timer for tuning
    Kokkos::Timer timer;
    double best_time = std::numeric_limits<double>::max();
    // first for hostspace
    // there is no tuning
    if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
      // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
      // for the innermost dimensions corresponds to the view extents
      best_tiling[rank-1] = end[rank-1] - start[rank-1];
      best_tiling[rank-2] = end[rank-2] - start[rank-2];
    }else {
      // for Cuda we need to tune the tiling
      const auto max_tile = policy.max_total_tile_size()/2;
      IndexArray<rank> current_tiling;
      IndexArray<rank> tile_one;
      for(index_t i = 0; i < rank; i++) {
        current_tiling[i] = 1;
        best_tiling[i] = 1;
        tile_one[i] = 1;
      }
      std::vector<index_t> fast_ind_tiles;
      index_t fast_ind = max_tile;
      while(fast_ind > 2) {
        fast_ind = fast_ind / 2;
        fast_ind_tiles.push_back(fast_ind);
      }
      for(auto &tile : fast_ind_tiles) {
        current_tiling = tile_one;
        current_tiling[0] = tile;
        index_t second_tile = max_tile / tile;
        while(second_tile > 1) {
          current_tiling[1] = second_tile;
          if(max_tile / tile / second_tile >=4 ){
            for(index_t i : {2, 1}) {
              current_tiling[2] = i;
              current_tiling[3] = i;
              auto tune_policy = Policy<rank>(start, end, current_tiling);
              double min_time = std::numeric_limits<double>::max();
              for(int ii = 0; ii < STREAM_NTIMES; ii++) {
                timer.reset();
                Kokkos::parallel_for(tune_policy, functor);
                Kokkos::fence();
                min_time = std::min(min_time, timer.seconds());
              }
              if(min_time < best_time) {
                best_time = min_time;
                best_tiling = current_tiling;
              }
              if (KLFT_VERBOSITY > 2) {
                printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                        current_tiling[0], current_tiling[1], 
                        current_tiling[2], current_tiling[3], min_time);
              }
            }
          }else if(max_tile / tile / second_tile == 2) {
            for(int64_t i : {2, 1}) {
              current_tiling[2] = i;
              current_tiling[3] = 1;
              auto tune_policy = Policy<rank>(start, end, current_tiling);
              double min_time = std::numeric_limits<double>::max();
              for(int ii = 0; ii < STREAM_NTIMES; ii++) {
                timer.reset();
                Kokkos::parallel_for(tune_policy, functor);
                Kokkos::fence();
                min_time = std::min(min_time, timer.seconds());
              }
              if(min_time < best_time) {
                best_time = min_time;
                best_tiling = current_tiling;
              }
              if (KLFT_VERBOSITY > 2) {
                printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                        current_tiling[0], current_tiling[1], 
                        current_tiling[2], current_tiling[3], min_time);
              }
            }
          }else {
            current_tiling[2] = 1;
            current_tiling[3] = 1;
            auto tune_policy = Policy<rank>(start, end, current_tiling);
            double min_time = std::numeric_limits<double>::max();
            for(int ii = 0; ii < STREAM_NTIMES; ii++) {
              timer.reset();
              Kokkos::parallel_for(tune_policy, functor);
              Kokkos::fence();
              min_time = std::min(min_time, timer.seconds());
            }
            if(min_time < best_time) {
              best_time = min_time;
              best_tiling = current_tiling;
            }
            if (KLFT_VERBOSITY > 2) {
              printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                      current_tiling[0], current_tiling[1], 
                      current_tiling[2], current_tiling[3], min_time);
            }
          }
          second_tile = second_tile / 2;
        }
      }
    }
    if (KLFT_VERBOSITY > 2) {
      printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
      printf("Best Time: %11.4e s\n", best_time);
    }
    // store the best tiling in the hash table
    if constexpr (rank == 4) {
      tuning_hash_table_4D.insert(functor_hash, best_tiling);
    } else if constexpr (rank == 3) {
      tuning_hash_table_3D.insert(functor_hash, best_tiling);
    } else if constexpr (rank == 2) {
      tuning_hash_table_2D.insert(functor_hash, best_tiling);
    }
    if (KLFT_VERBOSITY > 3) {
      double time_rec = std::numeric_limits<double>::max();
      auto tune_policy = Policy<rank>(start, end);
      for(int ii = 0; ii < STREAM_NTIMES; ii++) {
        timer.reset();
        Kokkos::parallel_for(tune_policy, functor);
        Kokkos::fence();
        time_rec = std::min(time_rec, timer.seconds());
      }
      printf("Time with default tile size: %11.4e s\n", time_rec);
      printf("Speedup: %f\n", time_rec / best_time);
    } 
    // run the kernel with the best tiling
    auto tune_policy = Policy<rank>(start, end, best_tiling);
    Kokkos::parallel_for(tune_policy, functor);
    Kokkos::fence();
    return;
  };

}