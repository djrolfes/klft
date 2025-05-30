#pragma once
#include "GLOBAL.hpp"
#include "FieldTypeHelper.hpp"
#include "Gauge_Util.hpp"


namespace klft {
    struct WilsonFlowParams{
        //define parameters for a given Wilson Flow calculation, the wilson flow follows arxiv.org/pdf/1006.4518

        // number of Wilson flow steps that should be generated
        index_t n_steps; 
        // flow time step size
        real_t eps; 
        // beta
        real_t beta;

        WilsonFlowParams(){
            // default parameters (eps = 0.01)
            n_steps = 10;
            eps = real_t(0.1/n_steps);
            beta = real_t(1.0);
        }
    };

template <size_t rank,
          size_t Nc,
          GaugeFieldKind Kind = GaugeFieldKind::Standard>
    struct WilsonFlow{
        // implement the Wilson flow, for now the field will not be copied, but it will be flown in place -> copying needs to be done before
        static_assert(rank == 4); // The wilson flow is only defined for 4D Fields
        WilsonFlowParams params;
        
        // get the correct deviceGaugeFieldType 
        using GaugeFieldT = typename DeviceGaugeFieldType<rank, Nc, Kind>::type;
        GaugeFieldT field;
        constGaugeField<rank, Nc> tmp_stap;
        GaugeField<rank, Nc> tmp_Z;
        index_t current_step {0};

        WilsonFlow() = delete;

        WilsonFlow(WilsonFlowParams &_params, GaugeFieldT &_field): params(_params), field(_field){
            const IndexArray<rank> dims = _field.dimensions;
            Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1], dims[2], dims[3]);
            Kokkos::realloc(Kokkos::WithoutInitializing, tmp_stap, dims[0], dims[1], dims[2], dims[3]);

//            Kokkos::deep_copy(_field.field, tmp_stap);
//            Kokkos::deep_copy(_field.field, tmp_Z);
            Kokkos::fence();
        }

        // execute the wilson flow 
        void flow(){ // todo: check this once by saving a staple field and once by locally calculating the staple
            for (int step = 0; step < params.n_steps; ++step){

                #pragma unroll
                for (index_t fstep = 0; fstep <3; ++fstep){
                    this->current_step = fstep;
                    tmp_stap = stapleField<rank, Nc, Kind>(field);
                    tune_and_launch_for<rank>("Wilsonflow-flow",IndexArray<rank>{0,0,0,0}, field.dimensions, *this);
                    Kokkos::fence();
                }

            }

        }

        template <typename indexType>
        KOKKOS_INLINE_FUNCTION void stepW1(indexType i0, indexType i1, indexType i2, indexType i3, index_t mu) const {
            SUN<Nc> Z0 = tmp_stap(i0, i1, i2, i3, mu) * conj(field(i0, i1, i2, i3, mu));
            tmp_Z(i0, i1, i2, i3, mu) = Z0;
            Z0 = make_antiherm(Z0 *(Nc/params.beta) * static_cast<real_t>(1.0 / 4.0));
		sun<Nc> tmp = from_Group(Z0);
            field(i0, i1, i2, i3, mu) = exp(tmp*params.eps) * field(i0, i1, i2, i3, mu);
            restoreSUN(field(i0, i1, i2, i3, mu));
        }

        template <typename indexType>
        KOKKOS_INLINE_FUNCTION void stepW2(indexType i0, indexType i1, indexType i2, indexType i3, index_t mu) const {
            SUN<Nc> Z1 = tmp_stap(i0, i1, i2, i3, mu) * conj(field(i0, i1, i2, i3, mu));
            SUN<Nc> Z0 = tmp_Z(i0, i1, i2, i3, mu);
            Z1 = Z1 * static_cast<real_t>(8.0 / 9.0) - Z0 * static_cast<real_t>(17.0 / 36.0);
            tmp_Z(i0, i1, i2, i3, mu) = Z1;
            Z1 = make_antiherm(Z1*(Nc/params.beta));
		sun<Nc> tmp = from_Group(Z1);
            field(i0, i1, i2, i3, mu) = exp(tmp*params.eps) * field(i0, i1, i2, i3, mu);
            restoreSUN(field(i0, i1, i2, i3, mu));
        }

        template <typename indexType>
        KOKKOS_INLINE_FUNCTION void stepV(indexType i0, indexType i1, indexType i2, indexType i3, index_t mu) const {
            SUN<Nc> Z2 = tmp_stap(i0, i1, i2, i3, mu) * conj(field(i0, i1, i2, i3, mu));
            SUN<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
            Z2 = (Z2 * static_cast<real_t>(3.0 / 2.0) - Z_old);
            Z2 = make_antiherm(Z2* (Nc/params.beta));
	    sun<Nc> tmp = from_Group(Z2);
            field(i0, i1, i2, i3, mu) = exp(tmp*params.eps) * field(i0, i1, i2, i3, mu);
            restoreSUN(field(i0, i1, i2, i3, mu));
        }

        template <typename indexType>
        KOKKOS_INLINE_FUNCTION void operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3) const {
            #pragma unroll
            for (index_t mu = 0; mu < 4; ++mu) {
                switch (this->current_step) {
                    case 0: stepW1(i0, i1, i2, i3, mu); break;
                    case 1: stepW2(i0, i1, i2, i3, mu); break;
                    case 2: stepV (i0, i1, i2, i3, mu); break;
                }
            }
        }
    };

}
