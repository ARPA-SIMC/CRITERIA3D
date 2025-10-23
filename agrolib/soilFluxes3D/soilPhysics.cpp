#include "soilPhysics.h"
#include "solver.h"
#include "otherFunctions.h"
#include <cassert>

using namespace soilFluxes3D::v2;
using namespace soilFluxes3D::v2::Math;

//Temp
#include "heat.h"
using namespace soilFluxes3D::v2::Heat;

namespace soilFluxes3D::v2
{
    extern __cudaMngd nodesData_t nodeGrid;
    extern __cudaMngd Solver* solver;
    extern __cudaMngd simulationFlags_t simulationFlags;
}

namespace soilFluxes3D::v2::Soil
{

    /*!
     * \brief Computes nodeIndex node volumetric water content as function of the node degree of saturation
     * \return theta (volumetric water content)     [m3 m-3]
     */
    __cudaSpec double computeNodeTheta(SF3Duint_t nodeIndex)
    {
        return computeNodeTheta_fromSe(nodeIndex, nodeGrid.waterData.saturationDegree[nodeIndex]);
    }

    /*!
     * \brief Computes nodeIndex node volumetric water content as function of degree of saturation
     * \return theta (volumetric water content)     [m3 m-3]
     */
    __cudaSpec double computeNodeTheta_fromSe(SF3Duint_t nodeIndex, double Se)
    {
        assert(!nodeGrid.surfaceFlag[nodeIndex]);   //TO DO: is needed?
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        return (Se * (nodeSoil.Theta_s - nodeSoil.Theta_r)) + nodeSoil.Theta_r;
    }


    /*!
     * \brief Computes nodeIndex node volumetric water content as function of signed water potential
     * \param signedPsi (signed water potential)    [m]
     * \return theta (volumetric water content)     [m3 m-3]
     */
    __cudaSpec double computeNodeTheta_fromSignedPsi(SF3Duint_t nodeIndex, double signedPsi)
    {
        if(nodeGrid.surfaceFlag[nodeIndex])
            return 1.;

        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        if(signedPsi >= 0.)
            return nodeSoil.Theta_s;

        return computeNodeTheta_fromSe(nodeIndex, computeNodeSe_fromPsi(nodeIndex, std::fabs(signedPsi)));
    }

    /*!
     * \brief Computes nodeIndex node degree of saturation as function of the node matric potential
     * \return Se (degree of saturation)     [-]
     */
    __cudaSpec double computeNodeSe(SF3Duint_t nodeIndex)
    {
        bool isSaturated = nodeGrid.waterData.pressureHead[nodeIndex] >= nodeGrid.z[nodeIndex];
        return isSaturated ? 1. : computeNodeSe_unsat(nodeIndex);
    }

    /*!
     * \brief Computes unsaturated nodeIndex node degree of saturation as function of the node matric potential
     * \return Se (degree of saturation)     [-]
     */
    __cudaSpec double computeNodeSe_unsat(SF3Duint_t nodeIndex)
    {
        double nodePsi = std::fabs(nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]);
        return computeNodeSe_fromPsi(nodeIndex, nodePsi);
    }

    /*!
     * \brief Computes unsaturated nodeIndex node degree of saturation as function of matric potential
     * \param psi (matric potential)        [m]
     * \return Se (degree of saturation)    [-]
     */
    __cudaSpec double computeNodeSe_fromPsi(SF3Duint_t nodeIndex, double psi)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);
        WRCModel model = solver->getWRCModel();
        switch(model)
        {
            case WRCModel::VanGenuchten:
                return std::pow(1. + std::pow(nodeSoil.VG_alpha * psi, nodeSoil.VG_n), -nodeSoil.VG_m);
                break;
            case WRCModel::ModifiedVanGenuchten:
                return (psi <= nodeSoil.VG_he) ? 1. : std::pow(1. + std::pow(nodeSoil.VG_alpha * psi, nodeSoil.VG_n), -nodeSoil.VG_m) * (1. / nodeSoil.VG_Sc);
                break;
            default:
                return noDataD;
        }
    }

    /*!
     * \brief Computes nodeIndex node degree of saturation as a function of input volumetric water content
     * \param theta (volumetric water content)      [m3 m-3]
     * \return Se (degree of saturation)     [-]
     */
    __cudaSpec double computeNodeSe_fromTheta(SF3Duint_t nodeIndex, double theta)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        if(theta >= nodeSoil.Theta_s)
            return 1.;

        if(theta < nodeSoil.Theta_r)
            return 0.;

        return (theta - nodeSoil.Theta_r) / (nodeSoil.Theta_s - nodeSoil.Theta_r);
    }

    /*!
     * \brief Computes nodeIndex node water potential as a function of node degree of saturation
     * \return psi (water potential)     [m]
     */
    __cudaSpec double computeNodePsi(SF3Duint_t nodeIndex)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        double temp;
        WRCModel model = solver->getWRCModel();
        switch(model)
        {
            case WRCModel::VanGenuchten:
                temp = std::pow(1. / nodeGrid.waterData.saturationDegree[nodeIndex], 1. / nodeSoil.VG_m) - 1.;
                break;
            case WRCModel::ModifiedVanGenuchten:
                temp = std::pow(1. / (nodeGrid.waterData.saturationDegree[nodeIndex] * nodeSoil.VG_Sc), 1. / nodeSoil.VG_m) - 1;
                break;
            default:
                return noDataD;
        }
        return (1. / nodeSoil.VG_alpha) * std::pow(temp, 1. / nodeSoil.VG_n);
    }

    /*!
     * \brief Computes nodeIndex node soil water total (liquid + vapor) conductivity
     * \return K (water conductivity)   [m s-1]
     */
    __cudaSpec double computeNodeK(SF3Duint_t nodeIndex)
    {
        double k = computeMualemSoilConductivity(*(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr), nodeGrid.waterData.saturationDegree[nodeIndex]);

        if(simulationFlags.computeHeat && simulationFlags.computeHeatVapor)
            k += computeNodeIsothermalVaporConductivity(nodeIndex, getNodeMeanTemperature(nodeIndex), nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]) * (GRAVITY / WATER_DENSITY);
    
        return k;
    }

    /*!
     * \brief Computes hydraulic conductivity as function of soil parameters and degree of saturation
     * \details K(Se) = Ksat * Se^L * {1 - [1 - Se^(1/m)]^m}^2
     * \warning very low values are possible (as e-12)
     * \return K (hydraulic conductivity)   [m s-1]
     */
    __cudaSpec double computeMualemSoilConductivity(soilData_t &soilData, double Se)
    {
        if(Se >= 1.)
            return soilData.K_sat;

        double temp, tNum, tDen;
        WRCModel model = solver->getWRCModel();
        switch(model)
        {
            case WRCModel::VanGenuchten:
                temp = 1. - std::pow(1. - std::pow(Se, 1. / soilData.VG_m), soilData.VG_m);
                break;
            case WRCModel::ModifiedVanGenuchten:
                tNum = 1. - std::pow(1. - std::pow(Se * soilData.VG_Sc, 1. / soilData.VG_m), soilData.VG_m);
                tDen = 1. - std::pow(1. - std::pow(soilData.VG_Sc, 1. / soilData.VG_m), soilData.VG_m);
                temp = tNum / tDen;
                break;
            default:
                return noDataD;
        }

        return soilData.K_sat * std::pow(Se, soilData.Mualem_L) * std::pow(temp, 2.);
    }

    /*!
     * \brief Compute the derivative of water volumetric content respect to the water potential
     * \details dTheta/dH = dSe/dH * (Theta_S - Theta_R) where
     *                      dSe/dH = -sgn(H-z) * alpha * n * m * [1 + (alpha * |H - z|)^n]^(-m-1) * (alpha * |H-z|)^(n-1)               if VanGenuchten
     *                  and dSe/dH = -sgn(H-z) * alpha * n * m * (1 / Sc) * [1 + (alpha * |H - z|)^n]^(-m-1) * (alpha * |H-z|)^(n-1)    if Modified VanGenuchten
     * \return dTheta/dH    [m-1]
     */
    __cudaSpec double computeNodedThetadH(SF3Duint_t nodeIndex)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        double psiCurr = std::fabs(SF3Dmin(0., nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]));
        double psiPrev = std::fabs(SF3Dmin(0., nodeGrid.waterData.oldPressureHead[nodeIndex] - nodeGrid.z[nodeIndex]));

        WRCModel model = solver->getWRCModel();
        switch(model)
        {
            case WRCModel::VanGenuchten:
                if((psiCurr == 0.) && (psiPrev == 0.))
                    return 0.;
                break;
            case WRCModel::ModifiedVanGenuchten:
                if((psiCurr <= nodeSoil.VG_he) && (psiPrev <= nodeSoil.VG_he))
                    return 0.;
                break;
            default:
                break;
        }

        double dSedH;
        if(psiCurr == psiPrev)
        {
            dSedH = nodeSoil.VG_alpha * nodeSoil.VG_n * nodeSoil.VG_m * std::pow(1. + std::pow(nodeSoil.VG_alpha * psiCurr, nodeSoil.VG_n), -(nodeSoil.VG_m + 1.)) * std::pow(nodeSoil.VG_alpha * psiCurr, nodeSoil.VG_n - 1.);
            if(model == WRCModel::ModifiedVanGenuchten)
                dSedH *= (1. / nodeSoil.VG_Sc);
        }
        else
        {
            double thetaCurr = computeNodeSe_fromPsi(nodeIndex, psiCurr);
            double thetaPrev = computeNodeSe_fromPsi(nodeIndex, psiPrev);
            dSedH = std::fabs((thetaCurr - thetaPrev) / (nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.waterData.oldPressureHead[nodeIndex]));
        }

        return dSedH * (nodeSoil.Theta_s - nodeSoil.Theta_r);
    }

    /*!
     * \brief Compute the derivative of the vapor volumetric content respect to the water potential
     * \details ...
     * \return dThetaV/dH    [m-1]
     */
    __cudaSpec double computeNodedThetaVdH(SF3Duint_t nodeIndex, double temperature, double dThetadH)
    {
        double h = nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex];
        double rH = computeSoilRelativeHumidity(h, temperature);

        double saturationVP = computeSaturationVaporPressure(temperature - ZEROCELSIUS);
        double saturationVC = computeVaporConcentration_fromPressure(saturationVP, temperature);

        double theta = computeNodeTheta_fromSignedPsi(nodeIndex, h);
        double dThetaVdPsi = (saturationVC * rH / WATER_DENSITY) * ((nodeGrid.soilSurfacePointers[nodeIndex].soilPtr->Theta_s - theta) * MH2O / (R_GAS * temperature) - dThetadH / GRAVITY);

        return dThetaVdPsi * GRAVITY;
    }

    /*!
     * \brief Return the nodeIndex node mean temperature
     * \return temperature    [K]
     */
    __cudaSpec double getNodeMeanTemperature(SF3Duint_t nodeIndex)
    {
        if(!simulationFlags.computeHeat)   //Add control over heat data
            return noDataD;

        return computeMean(nodeGrid.heatData.temperature[nodeIndex], nodeGrid.heatData.oldTemperature[nodeIndex], meanType_t::Arithmetic);
    }

    /*!
     * \brief Return the fraction of surface water
     * \return fraction of surface water    [-]
     */
    __cudaSpec double getNodeSurfaceWaterFraction(SF3Duint_t nodeIndex)
    {
        if(!nodeGrid.surfaceFlag[nodeIndex])
            return 0.;

        double hV = SF3Dmax(0., nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]);
        double h0 = SF3Dmax(0.001, nodeGrid.waterData.pond[nodeIndex]);

        return SF3Dmin(1., hV / h0);
    }

    __cudaSpec double nodeDistance2D(SF3Duint_t idx1, SF3Duint_t idx2)
    {
        double vec[] = {nodeGrid.x[idx1] - nodeGrid.x[idx2], nodeGrid.y[idx1] - nodeGrid.y[idx2]};
        return vectorNorm(vec, 2);
    }

    __cudaSpec double nodeDistance3D(SF3Duint_t idx1, SF3Duint_t idx2)
    {
        double vec[] = {nodeGrid.x[idx1] - nodeGrid.x[idx2], nodeGrid.y[idx1] - nodeGrid.y[idx2], nodeGrid.z[idx1] - nodeGrid.z[idx2]};
        return vectorNorm(vec, 3);
    }

} //namespace
