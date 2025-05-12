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

        WilsonFlowParams(){
            // default parameters (eps = 0.01)
            n_steps = 10;
            eps = real_t(0.1/n_steps);
        }
    };


    // TODO: rewrite to make the Wilsonflow act on a per link basis (not as much overhead from starting so many parallel dispatches.).
    template <size_t rank,
          size_t Nc,
          GaugeFieldKind Kind = GaugeFieldKind::Standard>
    struct WilsonFlow{
        // implement the Wilson flow, for now the field will not be copied, but it will be flown in place -> copying needs to be done before
        WilsonFlowParams params;
        
        // get the correct deviceGaugeFieldType 
        using GaugeFieldT = typename DeviceGaugeFieldType<rank, Nc, Kind>::type;
        GaugeFieldT field;
        constGaugeField<rank, Nc> tmp_stap;
        GaugeField<rank, Nc> tmp_Z;
        index_t current_step {0};

        WilsonFlow() = delete;

        WilsonFlow(WilsonFlowParams &_params, GaugeFieldT &_field){
            params = _params;
            field = _field;
            tmp_Z = _field;
            tmp_stap = _field;
        }

        // execute the wilson flow 
        void flow(){ // todo: check this once by saving a staple field and once by locally calculating the staple
            for (int step = 0; step < params.n_steps; ++step){
                #pragma unroll
                for (index_t fstep = 0; fstep <3; ++fstep){
                    this->current_step = fstep;
                    tmp_stap = Staple_Field(field);
                    tune_and_launch<Nd>("Wilsonflow-flow",IndexArray<Nd>{0}, field,dimensions, flow_step)
                    Kokkos::fence();
                }

            }

        }

        template <typename... Indices>
        KOKKOS_FORCEINLINE_FUNCTION void stepW1(const Indices... idxs, index_t mu, SUN<Nc> &Z0){
            Z0 = tmp_stap(idxs..., mu);
            Z0 *= conj(field(idxs..., mu));
            tmp_Z(idxs..., mu) = Z0;
            Z0*=(this->params.eps*static_cast<real_t>(1.0/4.0));         
        }

        template <typename... Indices>
        KOKKOS_FORCEINLINE_FUNCTION void stepW2(const Indices... idxs, index_t mu, SUN<Nc> &Z1){
            Z1 = tmp_stap(idxs..., mu);
            SUN<Nd> Z0 = tmp_Z(idxs..., mu);
            Z1 *= conj(field(idxs..., mu));
            Z1 = static_cast<real_t>(8.0/9.0) * Z1 - static_cast<real_t>(17/36) * Z0;
            tmp_Z(idxs..., mu) = Z1;
            Z1*=(this->params.eps);          
        }

        template <typename... Indices>
        KOKKOS_FORCEINLINE_FUNCTION void stepV(const Indices... idxs, index_t mu, SUN<Nc> &Z2){
            Z2 = tmp_stap(idxs..., mu);
            SUN<Nd> Z_old = tmp_Z(idxs..., mu);
            Z2 *= conj(field(idxs..., mu));
            Z2 = this->params.eps(static_cast<real_t>(3.0/2.0) * Z2 - Z_old);     
        }



        // do steps W1, W2 and V 
        template <typename... Indices>
        KOKKOS_FORCEINLINE_FUNCTION void flow_step(const Indices... idxs){
            SUN<Nc> Z;
            #pragma unroll
            for (index_t mu=0;mu<Nd;++mu){
                switch (this->current_step){
                    case 0: stepW1(idxs..., mu, Z); break;
                    case 1: stepW2(idxs..., mu, Z); break;
                    case 2: stepV (idxs..., mu, Z); break;
                }
                field(idxs..., mu) = exp(from_Group(Z));
            }
        }

    };
}