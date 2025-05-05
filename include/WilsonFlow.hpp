#pragma once
#include "GLOBAL.hpp"
#include "FieldTypeHelper.hpp"


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

    template <size_t rank,
          size_t Nc,
          GaugeFieldKind Kind = GaugeFieldKind::Standard>
    struct WilsonFlow{
        // implement the Wilson flow
        WilsonFlowParams params;
        
        // get the correct deviceGaugeFieldType 
        using GaugeFieldT = typename DeviceGaugeFieldType<rank, Nc, Kind>::type;
        GaugeFieldT field;

        

    };
}