#ifndef SOILFLUXES3D_SOIL_H
#define SOILFLUXES3D_SOIL_H

#include "macro.h"
#include "types_cpu.h"

namespace soilFluxes3D::Soil
{
    double computeNodeTheta(uint64_t nodeIndex);
    double computeNodeTheta_fromSe(uint64_t nodeIndex, double Se);
    double computeNodeTheta_fromSignedPsi(uint64_t nodeIndex, double signedPsi);

    double computeNodeSe(uint64_t nodeIndex);
    double computeNodeSe_unsat(uint64_t nodeIndex);
    double computeNodeSe_fromPsi(uint64_t nodeIndex, double psi);
    double computeNodeSe_fromTheta(uint64_t nodeIndex, double theta);

    double computeNodePsi(uint64_t nodeIndex);
    /*not used*/ double computeNodePsi_fromSe(uint64_t nodeIndex, double Se);

    double computeNodeK(uint64_t nodeIndex);
    double computeNodeK_Mualem(soilFluxes3D::New::soilData_t& soilData, double Se);

    double computeNodedThetadH(uint64_t nodeIndex);
    /*TO DO*/ double computeNodedThetaVdH(uint64_t nodeIndex, double temperature, double dThetadH);

    double getNodeMeanTemperature(uint64_t nodeIndex);


    double nodeDistance2D(uint64_t idx1, uint64_t idx2);
    double nodeDistance3D(uint64_t idx1, uint64_t idx2);
}

#endif // SOILFLUXES3D_SOIL_H
