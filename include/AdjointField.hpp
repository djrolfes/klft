#pragma once
#include "AdjointGroup.hpp"

namespace klft {

  template <typename T, class Adjoint, int Ndim = 4, int Nc = 3>
  class AdjointField {
  public:
    struct set_zero_s {};
    using DeviceView = Kokkos::View<T****>;

    static constexpr int adj_dim = Adjoint::nElements;


    DeviceView adjoint[Ndim][adj_dim];


    int LT,LX,LY,LZ;
    Kokkos::Array<int,4> dims;
    Kokkos::Array<int,4> max_dims;
    Kokkos::Array<int,4> array_dims;

    typedef Adjoint adjoint_group_t;

    AdjointField() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    AdjointField(const int &_LX, const int &_LY, const int &_LZ, const int &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      this->LT = _LT;
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    AdjointField(const Kokkos::Array<int,4> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    AdjointField(const int &_LX, const int &_LY, const int &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LT = _LT;
      this->LZ = 1;
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    AdjointField(const Kokkos::Array<int,3> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
      this->LZ = 1;
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    AdjointField(const int &_LX, const int &_LT) {
      this->LX = _LX;
      this->LT = _LT;
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    AdjointField(const Kokkos::Array<int,2> &_dims) {
      this->LX = _dims[0];
      this->LT = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < adj_dim; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
    }

    KOKKOS_INLINE_FUNCTION int get_Ndim() const { return Ndim; }

    KOKKOS_INLINE_FUNCTION int get_Nc() const { return Nc; }

    KOKKOS_INLINE_FUNCTION int get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

    KOKKOS_INLINE_FUNCTION int get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*(adj_dim); }

    KOKKOS_INLINE_FUNCTION int get_dim(const int &mu) const {
      return this->dims[mu];
    }

    KOKKOS_FUNCTION int get_max_dim(const int &mu) const {
      return this->max_dims[mu];
    }

    KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
      return this->array_dims[mu];
    }

    KOKKOS_INLINE_FUNCTION Adjoint get_adjoint(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Kokkos::Array<T,adj_dim> adj{};
      #pragma unroll
      for(int i = 0; i < adj_dim; i++) {
        adj[i] = this->adjoint[mu][i](x,y,z,t);
      }
      return Adjoint(adj);
    }

    KOKKOS_INLINE_FUNCTION Adjoint get_adjoint(const Kokkos::Array<int,4> &site, const int &mu) const {
      Kokkos::Array<T,adj_dim> adj{};
      #pragma unroll
      for(int i = 0; i < adj_dim; i++) {
        adj[i] = this->adjoint[mu][i](site[0],site[1],site[2],site[3]);
      }
      return Adjoint(adj);
    }

    KOKKOS_INLINE_FUNCTION void set_adjoint(const int &x, const int &y, const int &z, const int &t, const int &mu, const Adjoint &U) const {
      #pragma unroll
      for(int i = 0; i < adj_dim; i++) {
        this->adjoint[mu][i](x,y,z,t) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void set_adjoint(const Kokkos::Array<int,4> &site, const int &mu, const Adjoint &U) {
      #pragma unroll
      for(int i = 0; i < adj_dim; i++) {
        this->adjoint[mu][i](site[0],site[1],site[2],site[3]) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(set_zero_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      #pragma unroll
      for(int i = 0; i < adj_dim; i++) {
        this->adjoint[mu][i](x,y,z,t) = 0.0;
      }
    }

    void set_zero() {
      auto BulkPolicy = Kokkos::MDRangePolicy<set_zero_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
      Kokkos::parallel_for("set_zero", BulkPolicy, *this);
    }

  };

} // namespace klft