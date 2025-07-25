#include "soil_new.h"
#include "solver_new.h"
#include "otherFunctions.h"
#include <cassert>

using namespace soilFluxes3D::New;
using namespace soilFluxes3D::Math;

extern nodesData_t nodeGrid;
extern Solver& solver;
extern simulationFlags_t simulationFlags;

namespace soilFluxes3D::Soil
{

    /*!
     * \brief Computes nodeIndex node volumetric water content as function of the node degree of saturation
     * \return theta (volumetric water content)     [m3 m-3]
     */
    double computeNodeTheta(uint64_t nodeIndex)
    {
        assert(!nodeGrid.surfaceFlag[nodeIndex]);   //TO DO: is needed?
        return (nodeGrid.waterData.saturationDegree[nodeIndex] * (nodeGrid.soilSurfacePointers->soilPtr->Theta_s - nodeGrid.soilSurfacePointers->soilPtr->Theta_r)) + nodeGrid.soilSurfacePointers->soilPtr->Theta_r;
    }

    /*!
     * \brief Computes nodeIndex node volumetric water content as function of the node signed water potential
     * \param signedPsi (signed water potential)    [m]
     * \return theta (volumetric water content)     [m3 m-3]
     */
    double computeNodeTheta_fromSignedPsi(uint64_t nodeIndex, double signedPsi)
    {
        if(nodeGrid.surfaceFlag[nodeIndex])
            return 1.;

        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        if(signedPsi >= 0.)
            return nodeSoil.Theta_s;

        return computeNodeTheta_fromSe(nodeIndex, computeNodeSe_fromPsi(nodeIndex, fabs(signedPsi)));
    }

    /*!
     * \brief Computes nodeIndex node degree of saturation as function of the node matric potential
     * \return Se (degree of saturation)     [-]
     */
    double computeNodeSe(uint64_t nodeIndex)
    {
        bool isSaturated = nodeGrid.waterData.pressureHead[nodeIndex] >= nodeGrid.z[nodeIndex];
        return isSaturated ? 1. : computeNodeSe_unsat(nodeIndex);
    }

    /*!
     * \brief Computes unsaturated nodeIndex node degree of saturation as function of the node matric potential
     * \return Se (degree of saturation)     [-]
     */
    double computeNodeSe_unsat(uint64_t nodeIndex)
    {
        double nodePsi = fabs(nodeGrid.waterData.pressureHead[nodeIndex] - nodeGrid.z[nodeIndex]);
        return computeNodeSe_fromPsi(nodeIndex, nodePsi);
    }

    /*!
     * \brief Computes unsaturated nodeIndex node degree of saturation as function of matric potential
     * \param psi (matric potential)        [m]
     * \return Se (degree of saturation)    [-]
     */
    double computeNodeSe_fromPsi(uint64_t nodeIndex, double psi)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        switch(solver.getWRCModel())
        {
            case VanGenuchten:
                return pow(1. + pow(nodeSoil.VG_alpha * psi, nodeSoil.VG_n), -nodeSoil.VG_m);
                break;
            case ModifiedVanGenuchten:
                return (psi <= nodeSoil.VG_he) ? 1. : pow(1. + pow(nodeSoil.VG_alpha * psi, nodeSoil.VG_n), -nodeSoil.VG_m) * (1. / nodeSoil.VG_Sc);
                break;
            default:
                return noData;
        }
    }

    /*!
     * \brief Computes nodeIndex node degree of saturation as a function of input volumetric water content
     * \param theta (volumetric water content)      [m3 m-3]
     * \return Se (degree of saturation)     [-]
     */
    double computeNodeSe_fromTheta(uint64_t nodeIndex, double theta)
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
    double computeNodePsi(uint64_t nodeIndex)
    {
        soilData_t& nodeSoil = *(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr);

        double temp;
        switch(solver.getWRCModel())
        {
            case VanGenuchten:
                temp = pow(1. / nodeGrid.waterData.saturationDegree[nodeIndex], 1. / nodeSoil.VG_m) - 1.;
                break;
            case ModifiedVanGenuchten:
                temp = pow(1. / (nodeGrid.waterData.saturationDegree[nodeIndex] * nodeSoil.VG_Sc), 1. / nodeSoil.VG_m) - 1;
                break;
            default:
                return noData;
        }
        return (1. / nodeSoil.VG_alpha) * pow(temp, 1. / nodeSoil.VG_n);
    }

    /*!
     * \brief Computes nodeIndex node soil water total (liquid + vapor) conductivity
     * \return K (water conductivity)   [m s-1]
     */
    double computeNodeK(uint64_t nodeIndex)
    {
        double k = computeNodeK_Mualem(*(nodeGrid.soilSurfacePointers[nodeIndex].soilPtr), nodeGrid.waterData.saturationDegree[nodeIndex]);

        if(simulationFlags.computeHeat && simulationFlags.computeHeatVapor)
        {
            //TO DO: Heat
        }

        return k;
    }

    /*!
     * \brief Computes hydraulic conductivity as function of soil parameters and degree of saturation
     * \details K(Se) = Ksat * Se^L * {1 - [1 - Se^(1/m)]^m}^2
     * \warning very low values are possible (as e-12)
     * \return K (water conductivity)   [m s-1]
     */
    double computeNodeK_Mualem(soilData_t &soilData, double Se)
    {
        if(Se >= 1.)
            return soilData.K_sat;

        double temp, tNum, tDen;
        switch(solver.getWRCModel())
        {
            case VanGenuchten:
                temp = 1. - pow(1. - pow(Se, 1 / soilData.VG_m), soilData.VG_m);
                break;
            case ModifiedVanGenuchten:
                tNum = 1. - pow(1. - pow(Se * soilData.VG_Sc, 1 / soilData.VG_m), soilData.VG_m);
                tDen = 1. - pow(1. - pow(soilData.VG_Sc, 1 / soilData.VG_m), soilData.VG_m);
                temp = tNum / tDen;
                break;
            default:
                return noData;
        }

        return soilData.K_sat * pow(Se, soilData.Mualem_L) * pow(temp, 2.);
    }

    double getNodeMeanTemperature(uint64_t nodeIndex)
    {
        if(!simulationFlags.computeHeat)   //Add control over heat data
            return noData;

        return computeMean(nodeGrid.heatData.temperature[nodeIndex], nodeGrid.heatData.oldTemperature[nodeIndex], Arithmetic);
    }


    double nodeDistance2D(uint64_t idx1, uint64_t idx2)
    {
        double vec[] = {nodeGrid.x[idx1] - nodeGrid.x[idx2], nodeGrid.y[idx1] - nodeGrid.y[idx2]};
        return vectorNorm(vec, 2);
    }

    double nodeDistance3D(uint64_t idx1, uint64_t idx2)
    {
        double vec[] = {nodeGrid.x[idx1] - nodeGrid.x[idx2], nodeGrid.y[idx1] - nodeGrid.y[idx2], nodeGrid.z[idx1] - nodeGrid.z[idx2]};
        return vectorNorm(vec, 3);
    }


} //namespace
