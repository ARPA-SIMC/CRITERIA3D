#pragma once

#include "macro.h"
#include "types.h"

using namespace soilFluxes3D::v2;

namespace soilFluxes3D::v2::Soil
{
    __cudaSpec double computeNodeTheta(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeTheta_fromSe(SF3Duint_t nodeIndex, double Se);
    __cudaSpec double computeNodeTheta_fromSignedPsi(SF3Duint_t nodeIndex, double signedPsi);

    __cudaSpec double computeNodeSe(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeSe_unsat(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodeSe_fromPsi(SF3Duint_t nodeIndex, double psi);
    __cudaSpec double computeNodeSe_fromTheta(SF3Duint_t nodeIndex, double theta);

    __cudaSpec double computeNodePsi(SF3Duint_t nodeIndex);
    /*not used*/ double computeNodePsi_fromSe(SF3Duint_t nodeIndex, double Se);

    __cudaSpec double computeNodeK(SF3Duint_t nodeIndex);
    __cudaSpec double computeMualemSoilConductivity(soilData_t& soilData, double Se);

    __cudaSpec double computeNodedThetadH(SF3Duint_t nodeIndex);
    __cudaSpec double computeNodedThetaVdH(SF3Duint_t nodeIndex, double temperature, double dThetadH);

    __cudaSpec double getNodeMeanTemperature(SF3Duint_t nodeIndex);

    __cudaSpec double getNodeSurfaceWaterFraction(SF3Duint_t nodeIndex);

    __cudaSpec double nodeDistance2D(SF3Duint_t idx1, SF3Duint_t idx2);
    __cudaSpec double nodeDistance3D(SF3Duint_t idx1, SF3Duint_t idx2);
}
