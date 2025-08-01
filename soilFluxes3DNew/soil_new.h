#ifndef SOILFLUXES3D_SOIL_H
#define SOILFLUXES3D_SOIL_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Soil
{
    __cudaSpec double computeNodeTheta(uint64_t nodeIndex);
    __cudaSpec double computeNodeTheta_fromSe(uint64_t nodeIndex, double Se);
    __cudaSpec double computeNodeTheta_fromSignedPsi(uint64_t nodeIndex, double signedPsi);

    __cudaSpec double computeNodeSe(uint64_t nodeIndex);
    __cudaSpec double computeNodeSe_unsat(uint64_t nodeIndex);
    __cudaSpec double computeNodeSe_fromPsi(uint64_t nodeIndex, double psi);
    __cudaSpec double computeNodeSe_fromTheta(uint64_t nodeIndex, double theta);

    __cudaSpec double computeNodePsi(uint64_t nodeIndex);
    /*not used*/ double computeNodePsi_fromSe(uint64_t nodeIndex, double Se);

    __cudaSpec double computeNodeK(uint64_t nodeIndex);
    __cudaSpec double computeNodeK_Mualem(soilFluxes3D::New::soilData_t& soilData, double Se);

    __cudaSpec double computeNodedThetadH(uint64_t nodeIndex);
    /*TO DO*/ __cudaSpec double computeNodedThetaVdH(uint64_t nodeIndex, double temperature, double dThetadH);

    __cudaSpec double getNodeMeanTemperature(uint64_t nodeIndex);


    __cudaSpec double nodeDistance2D(uint64_t idx1, uint64_t idx2);
    __cudaSpec double nodeDistance3D(uint64_t idx1, uint64_t idx2);
}

#endif // SOILFLUXES3D_SOIL_H
