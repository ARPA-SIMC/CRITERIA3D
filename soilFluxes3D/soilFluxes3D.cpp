/*!
    \name soilFluxes3D.cpp
    \copyright (C) 2011 Fausto Tomei, Gabriele Antolini, Antonio Volta,
                        Alberto Pistocchi, Marco Bittelli

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "physics.h"
#include "header/types.h"
#include "header/memory.h"
#include "header/soilPhysics.h"
#include "header/soilFluxes3D.h"
#include "header/solver.h"
#include "header/balance.h"
#include "header/water.h"
#include "header/boundary.h"
#include "header/heat.h"
#include "header/extra.h"

/*! global variables */

TParameters myParameters;
TCrit3DStructure myStructure;

Tculvert myCulvert;

TCrit3Dnode *myNode = nullptr;
TmatrixElement **A = nullptr;

double *invariantFlux = nullptr;
double *C = nullptr;
double *X = nullptr;
double *b = nullptr;

std::vector< std::vector<Tsoil>> Soil_List;
std::vector<Tsoil> Surface_List;


namespace soilFluxes3D {

int DLL_EXPORT __STDCALL test()
{
    return(CRIT3D_OK);
}

void DLL_EXPORT __STDCALL cleanMemory()
{
    cleanNodes();
    cleanArrays();
    // TODO clean balance
}

void DLL_EXPORT __STDCALL initializeHeat(short myType, bool computeAdvectiveHeat, bool computeLatentHeat)
{
    myStructure.saveHeatFluxesType = myType;
    myStructure.computeHeatAdvection = computeAdvectiveHeat;
    myStructure.computeHeatVapor = computeLatentHeat;
}


int DLL_EXPORT __STDCALL initialize(long nrNodes, int nrLayers, int nrLateralLinks,
                                    bool computeWater_, bool computeHeat_, bool computeSolutes_)
{
    /*! clean the old data structures */
    cleanMemory();

    myParameters.initialize();
    myStructure.initialize();   

    myStructure.computeWater = computeWater_;
    myStructure.computeHeat = computeHeat_;
    if (computeHeat_)
    {
        myStructure.computeHeatVapor = true;
        myStructure.computeHeatAdvection = true;
    }
    myStructure.computeSolutes = computeSolutes_;

    myStructure.nrNodes = nrNodes;
    myStructure.nrLayers = nrLayers;
    myStructure.nrLateralLinks = nrLateralLinks;
    /*! max nr columns = nr. of lateral links + 2 columns for up and down link + 1 column for diagonal */
    myStructure.maxNrColumns = nrLateralLinks + 2 + 1;

    /*! build the nodes vector */
    myNode = (TCrit3Dnode *) calloc(myStructure.nrNodes, sizeof(TCrit3Dnode));
	for (long i = 0; i < myStructure.nrNodes; i++)
	{
		myNode[i].Soil = nullptr;
		myNode[i].boundary = nullptr;
		myNode[i].up.index = NOLINK;
        myNode[i].down.index = NOLINK;

        myNode[i].lateral = (TlinkedNode *) calloc(myStructure.nrLateralLinks, sizeof(TlinkedNode));

        for (short l = 0; l < myStructure.nrLateralLinks; l++)
        {
            myNode[i].lateral[l].index = NOLINK;
            if (myStructure.computeHeat || myStructure.computeSolutes)
                myNode[i].lateral[l].linkedExtra = new(TCrit3DLinkedNodeExtra);
        }
    }

    /*! build the matrix */
    if (myNode == nullptr)
        return MEMORY_ERROR;
    else
        return initializeArrays();
 }


/*!
   \brief setNumericalParameters
   \return numerical solution parameters
*/
int DLL_EXPORT __STDCALL setNumericalParameters(float minDeltaT, float maxDeltaT, int maxIterationNumber,
                        int maxApproximationsNumber, int ResidualTolerance, float MBRThreshold)
{
    if (minDeltaT < 0.1f) minDeltaT = 0.1f;
    if (minDeltaT > 3600) minDeltaT = 3600;
    myParameters.delta_t_min = double(minDeltaT);

    if (maxDeltaT < 60) maxDeltaT = 60;
    if (maxDeltaT > 3600) maxDeltaT = 3600;
    if (maxDeltaT < minDeltaT) maxDeltaT = minDeltaT;
    myParameters.delta_t_max = double(maxDeltaT);

    myParameters.current_delta_t = myParameters.delta_t_max;

    if (maxIterationNumber < 10) maxIterationNumber = 10;
    if (maxIterationNumber > MAX_NUMBER_ITERATIONS) maxIterationNumber = MAX_NUMBER_ITERATIONS;
    myParameters.iterazioni_max = maxIterationNumber;

    if (maxApproximationsNumber < 1) maxApproximationsNumber = 1;

    if (maxApproximationsNumber > MAX_NUMBER_APPROXIMATIONS)
            maxApproximationsNumber = MAX_NUMBER_APPROXIMATIONS;

    myParameters.maxApproximationsNumber = maxApproximationsNumber;

    if (ResidualTolerance < 4) ResidualTolerance = 4;
    if (ResidualTolerance > 16) ResidualTolerance = 16;
    myParameters.ResidualTolerance = pow(double(10.), -ResidualTolerance);

    if (MBRThreshold < 1) MBRThreshold = 1;
    if (MBRThreshold > 6) MBRThreshold = 6;
    myParameters.MBRThreshold = pow(double(10.), double(-MBRThreshold));

    return CRIT3D_OK;
}


/*!
 * \brief setHydraulicProperties
 *  default values:
 *  waterRetentionCurve = MODIFIED_VANGENUCHTEN
 *  meanType = MEAN_LOGARITHMIC
 *  k_lateral_vertical_ratio = 10
 * \param waterRetentionCurve
 * \param conductivityMeanType
 * \param horizVertRatioConductivity
 * \return OK or PARAMETER_ERROR
 */
int DLL_EXPORT __STDCALL setHydraulicProperties(int waterRetentionCurve,
                        int conductivityMeanType, float horizVertRatioConductivity)
{
    myParameters.waterRetentionCurve = waterRetentionCurve;
    myParameters.meanType = conductivityMeanType;

    if  ((horizVertRatioConductivity >= 1) && (horizVertRatioConductivity <= 100))
    {
        myParameters.k_lateral_vertical_ratio = horizVertRatioConductivity;
        return CRIT3D_OK;
    }
    else
    {
        myParameters.k_lateral_vertical_ratio = 10.;
        return PARAMETER_ERROR;
    }
}

/*!
 * \brief setNode
 * \param myIndex
 * \param x
 * \param y
 * \param z
 * \param volume_or_area
 * \param isSurface
 * \param isBoundary
 * \param boundaryType
 * \param slope
 * \return node position and properties
 */
 int DLL_EXPORT __STDCALL setNode(long myIndex, float x, float y, double z, double volume_or_area, bool isSurface,
                        bool isBoundary, int boundaryType, float slope, float boundaryArea)
 {
    if (myNode == nullptr) return(MEMORY_ERROR);
    if ((myIndex < 0) || (myIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

	if (isBoundary)
	{
		myNode[myIndex].boundary = new(Tboundary);
        initializeBoundary(myNode[myIndex].boundary, boundaryType, slope, boundaryArea);
	}

    if ((myStructure.computeHeat || myStructure.computeSolutes) && ! isSurface)
    {
        myNode[myIndex].extra = new(TCrit3DnodeExtra);
        initializeExtra(myNode[myIndex].extra, myStructure.computeHeat, myStructure.computeSolutes);
    }

    myNode[myIndex].x = x;
    myNode[myIndex].y = y;
    myNode[myIndex].z = z;
    myNode[myIndex].volume_area = volume_or_area;   /*!< area on surface elements, volume on sub-surface */

	myNode[myIndex].isSurface = isSurface;

    myNode[myIndex].waterSinkSource = 0.;

    return CRIT3D_OK;
 }


 int DLL_EXPORT __STDCALL setNodeLink(long n, long linkIndex, short direction, float interfaceArea)
 {
    /*! error check */
    if (myNode == nullptr) return MEMORY_ERROR;

    if ((n < 0) || (n >= myStructure.nrNodes) || (linkIndex < 0) || (linkIndex >= myStructure.nrNodes))
        return INDEX_ERROR;

    short j;
    switch (direction)
    {
        case UP :
                    myNode[n].up.index = linkIndex;
                    myNode[n].up.area = interfaceArea;
                    myNode[n].up.sumFlow = 0;

                    if (myStructure.computeHeat || myStructure.computeSolutes)
                    {
                        myNode[n].up.linkedExtra = new(TCrit3DLinkedNodeExtra);
                        initializeLinkExtra(myNode[n].up.linkedExtra, myStructure.computeHeat, myStructure.computeSolutes);
                    }

                    break;
        case DOWN :
                    myNode[n].down.index = linkIndex;
                    myNode[n].down.area = interfaceArea;
					myNode[n].down.sumFlow = 0;

                    if (myStructure.computeHeat || myStructure.computeSolutes)
                    {
                        myNode[n].down.linkedExtra = new(TCrit3DLinkedNodeExtra);
                        initializeLinkExtra(myNode[n].down.linkedExtra, myStructure.computeHeat, myStructure.computeSolutes);
                    }

                    break;
        case LATERAL :
                    j = 0;
                    while ((j < myStructure.nrLateralLinks) && (myNode[n].lateral[j].index != NOLINK)) j++;
                    if (j == myStructure.nrLateralLinks) return (TOPOGRAPHY_ERROR);
                    myNode[n].lateral[j].index = linkIndex;
                    myNode[n].lateral[j].area = interfaceArea;
					myNode[n].lateral[j].sumFlow = 0;

                    if (myStructure.computeHeat || myStructure.computeSolutes)
                    {
                        myNode[n].lateral[j].linkedExtra = new(TCrit3DLinkedNodeExtra);
                        initializeLinkExtra(myNode[n].lateral[j].linkedExtra, myStructure.computeHeat, myStructure.computeSolutes);
                    }

                    break;
        default :
                    return PARAMETER_ERROR;
    }
    return CRIT3D_OK;
 }


 int DLL_EXPORT __STDCALL setCulvert(long nodeIndex, double roughness, double slope, double width, double height)
 {
	 if ((nodeIndex < 0) || (!myNode[nodeIndex].isSurface))
	 {
		 myCulvert.index = NOLINK;
		 return(INDEX_ERROR);
	 }

	 myCulvert.index = nodeIndex;
	 myCulvert.roughness = roughness;			// [s m^-1/3]
	 myCulvert.slope = slope;					// [-]
	 myCulvert.width = width;					// [m]
	 myCulvert.height = height;					// [m]

	myNode[nodeIndex].boundary = new(Tboundary);
    double boundaryArea = width*height;
    initializeBoundary(myNode[nodeIndex].boundary, BOUNDARY_CULVERT, float(slope), float(boundaryArea));

	 return(CRIT3D_OK);
 }


/*!
 * \brief setNodeSurface
 * \param nodeIndex
 * \param surfaceIndex
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setNodeSurface(long nodeIndex, int surfaceIndex)
 {
    if (myNode == nullptr)
        return MEMORY_ERROR;
    if (nodeIndex < 0 || (! myNode[nodeIndex].isSurface))
        return INDEX_ERROR;
    if (surfaceIndex < 0 || surfaceIndex >= int(Surface_List.size()))
        return PARAMETER_ERROR;

    myNode[nodeIndex].Soil = &Surface_List[surfaceIndex];

    return(CRIT3D_OK);
 }


/*!
 * \brief setNodeSoil
 * \param nodeIndex
 * \param soilIndex
 * \param horizonIndex
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setNodeSoil(long nodeIndex, int soilIndex, int horizonIndex)
 {
    if (myNode == nullptr)
        return MEMORY_ERROR;
    if (nodeIndex < 0 || nodeIndex >= myStructure.nrNodes)
        return INDEX_ERROR;
    if (soilIndex < 0 || soilIndex >= int(Soil_List.size()))
        return PARAMETER_ERROR;
    if (horizonIndex < 0 || horizonIndex >= int(Soil_List[soilIndex].size()))
        return PARAMETER_ERROR;

    myNode[nodeIndex].Soil = &Soil_List[soilIndex][horizonIndex];

    return CRIT3D_OK;
 }


/*!
 * \brief setSoilProperties
 * \param nSoil
 * \param nHorizon
 * \param VG_alpha  [m-1]       Van Genutchen alpha parameter (warning: usually is kPa-1 in literature)
 * \param VG_n      [-]         Van Genutchen n parameter ]1, 10]
 * \param VG_m
 * \param VG_he
 * \param ThetaR    [m3 m-3]    residual water content
 * \param ThetaS    [m3 m-3]    saturated water content
 * \param Ksat      [m s-1]     saturated hydraulic conductivity
 * \param L         [-]         tortuosity (Mualem equation)
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setSoilProperties(int nSoil, int nHorizon, double VG_alpha, double VG_n, double VG_m,
                        double VG_he, double ThetaR, double ThetaS, double Ksat, double L, double organicMatter, double clay)
 {

    if (VG_alpha <= 0 || ThetaR < 0 || ThetaR >= 1 || ThetaS <= 0 || ThetaS > 1 || ThetaR > ThetaS)
        return PARAMETER_ERROR;

    if (nSoil >= int(Soil_List.size()))
        Soil_List.resize(nSoil+1);
    if (nHorizon >= int(Soil_List[nSoil].size()))
        Soil_List[nSoil].resize(nHorizon+1);

    Soil_List[nSoil][nHorizon].VG_alpha = VG_alpha;
    Soil_List[nSoil][nHorizon].VG_n = VG_n;
    Soil_List[nSoil][nHorizon].VG_m = VG_m;

    Soil_List[nSoil][nHorizon].VG_he = VG_he;
    Soil_List[nSoil][nHorizon].VG_Sc = pow(1. + pow(VG_alpha * VG_he, VG_n), -VG_m);

    Soil_List[nSoil][nHorizon].Theta_r = ThetaR;
    Soil_List[nSoil][nHorizon].Theta_s = ThetaS;
    Soil_List[nSoil][nHorizon].K_sat = Ksat;
    Soil_List[nSoil][nHorizon].Mualem_L = L;

    Soil_List[nSoil][nHorizon].organicMatter = organicMatter;
    Soil_List[nSoil][nHorizon].clay = clay;

    return CRIT3D_OK;
 }


 int DLL_EXPORT __STDCALL setSurfaceProperties(int surfaceIndex, double roughness, double surfacePond)
 {
    if (roughness < 0 || surfacePond < 0)
        return PARAMETER_ERROR;

    if (surfaceIndex > int(Surface_List.size()-1))
        Surface_List.resize(surfaceIndex+1);

    Surface_List[surfaceIndex].Roughness = roughness;
    Surface_List[surfaceIndex].Pond = surfacePond;

    return CRIT3D_OK;
 }


/*!
 * \brief setMatricPotential
 * \param nodeIndex
 * \param potential [m]
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setMatricPotential(long nodeIndex, double potential)
 {
     if (myNode == nullptr)
         return MEMORY_ERROR;
     if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
         return INDEX_ERROR;

     myNode[nodeIndex].H = potential + myNode[nodeIndex].z;
     myNode[nodeIndex].oldH = myNode[nodeIndex].H;

     if (myNode[nodeIndex].isSurface)
     {
         myNode[nodeIndex].Se = 1.;
         myNode[nodeIndex].k = NODATA;
     }
     else
     {
         myNode[nodeIndex].Se = computeSe(nodeIndex);
         myNode[nodeIndex].k = computeK(nodeIndex);
     }

     return CRIT3D_OK;
 }


    /*!
     * \brief setToalPotential
     * \param nodeIndex
     * \param totalPotential [m]
     * \return OK/ERROR
     */
	int DLL_EXPORT __STDCALL setTotalPotential(long nodeIndex, double totalPotential)
 {

	 if (myNode == nullptr)
		 return(MEMORY_ERROR);

	 if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
		 return(INDEX_ERROR);

     myNode[nodeIndex].H = totalPotential;
	 myNode[nodeIndex].oldH = myNode[nodeIndex].H;

	 if (myNode[nodeIndex].isSurface)
	 {
		 myNode[nodeIndex].Se = 1.;
		 myNode[nodeIndex].k = NODATA;
	 }
	 else
	 {
         myNode[nodeIndex].Se = computeSe(nodeIndex);
         myNode[nodeIndex].k = computeK(nodeIndex);
	 }

	 return(CRIT3D_OK);
 }


/*!
 * \brief setWaterContent
 * \param nodeIndex
 * \param waterContent [m^3 m^-3]
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setWaterContent(long nodeIndex, double waterContent)
 {
    if (myNode == nullptr) return MEMORY_ERROR;

    if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return INDEX_ERROR;

    if (waterContent < 0.) return PARAMETER_ERROR;

    if (myNode[nodeIndex].isSurface)
            {
            /*! surface */
            myNode[nodeIndex].H = myNode[nodeIndex].z + waterContent;
            myNode[nodeIndex].oldH = myNode[nodeIndex].H;
            myNode[nodeIndex].Se = 1.;
            myNode[nodeIndex].k = 0.;
            }
    else
            {
            if (waterContent > 1.0) return PARAMETER_ERROR;
            myNode[nodeIndex].Se = Se_from_theta(nodeIndex, waterContent);
            myNode[nodeIndex].H = myNode[nodeIndex].z - psi_from_Se(nodeIndex);
            myNode[nodeIndex].oldH = myNode[nodeIndex].H;
            myNode[nodeIndex].k = computeK(nodeIndex);
            }

    return CRIT3D_OK;
 }


/*!
 * \brief setWaterSinkSource
 * \param nodeIndex
 * \param waterSinkSource [m^3/sec] flow
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setWaterSinkSource(long nodeIndex, double waterSinkSource)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return INDEX_ERROR;

    myNode[nodeIndex].waterSinkSource = waterSinkSource;

    return CRIT3D_OK;
 }


/*!
 * \brief setPrescribedTotalPotential
 * \param nodeIndex
 * \param prescribedTotalPotential [m]
 * \return OK/ERROR
 */
 int DLL_EXPORT __STDCALL setPrescribedTotalPotential(long nodeIndex, double prescribedTotalPotential)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return INDEX_ERROR;
    if (myNode[nodeIndex].boundary == nullptr) return BOUNDARY_ERROR;
    if (myNode[nodeIndex].boundary->type != BOUNDARY_PRESCRIBEDTOTALPOTENTIAL) return BOUNDARY_ERROR;

    myNode[nodeIndex].boundary->prescribedTotalPotential = prescribedTotalPotential;

    return CRIT3D_OK;
 }


/*!
 * \brief getWaterContent
 * \param nodeIndex
 * \return water content at the surface: [m] surface water level; and sub-surface: [m^3 m^-3] volumetric water content
 */
 double DLL_EXPORT __STDCALL getWaterContent(long nodeIndex)
 {
        if (myNode == nullptr) return(MEMORY_ERROR);
        if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

        if  (myNode[nodeIndex].isSurface)
            /*! surface */
            return (myNode[nodeIndex].H - myNode[nodeIndex].z);
        else
            /*! sub-surface */
            return (theta_from_Se(nodeIndex));
 }


/*!
 * \brief getAvailableWaterContent
 * \param index
 * \return  available water content (over wilting point) at surface: water level [m]; sub-surface: awc [m^3 m^-3]
 */
 double DLL_EXPORT __STDCALL getAvailableWaterContent(long index)
 {
        if (myNode == nullptr) return(MEMORY_ERROR);
        if ((index < 0) || (index >= myStructure.nrNodes)) return(INDEX_ERROR);

        if  (myNode[index].isSurface)
            /*! surface */
            return (myNode[index].H - myNode[index].z);
        else
            /*! sub-surface */
            return MAXVALUE(0.0, theta_from_Se(index) - theta_from_sign_Psi(-160, index));
 }


    /*!
     * \brief getWaterDeficit
     * \param index
     * \param fieldCapacity
     * \return water deficit at surface: 0; sub-surface: [m^3 m^-3]
     */
	double DLL_EXPORT __STDCALL getWaterDeficit(long index, double fieldCapacity)
 {
        if (myNode == nullptr) return(MEMORY_ERROR);
        if ((index < 0) || (index >= myStructure.nrNodes)) return(INDEX_ERROR);

        if  (myNode[index].isSurface)
            /*! surface */
            return (0.0);
        else
            /*! sub-surface */
            return (theta_from_sign_Psi(-fieldCapacity, index) - theta_from_Se(index));
 }



/*!
  * \brief getDegreeOfSaturation
  * \param nodeIndex
  * \return degree of saturation at surface: water presence 0-100 [-]; sub-surface: degree of saturation [%]
  */
 double DLL_EXPORT __STDCALL getDegreeOfSaturation(long nodeIndex)
 {
        if (myNode == nullptr) return(MEMORY_ERROR);
        if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

        if  (myNode[nodeIndex].isSurface)
        {
            if ((myNode[nodeIndex].H - myNode[nodeIndex].z) > 0.0001)
                return 100;
            else
                return 0;
        }
        else
            return (myNode[nodeIndex].Se*100.0);
 }


 /*!
  * \brief getTotalWaterContent
  * \return total water content [m^3]
  */
 double DLL_EXPORT __STDCALL getTotalWaterContent()
 {
    return(computeTotalWaterContent());
 }


 /*!
  * \brief getWaterConductivity
  * \param nodeIndex
  * \return hydraulic conductivity (k) [m/s]
  */
 double DLL_EXPORT __STDCALL getWaterConductivity(long nodeIndex)
 {
    /*! error check */
    if (myNode == nullptr) return(MEMORY_ERROR);
    if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

    return (myNode[nodeIndex].k);
 }


 /*!
  * \brief getMatricPotential
  * \param nodeIndex [-]
  * \return matric potential [m]
  */
 double DLL_EXPORT __STDCALL getMatricPotential(long nodeIndex)
 {
    if (myNode == nullptr) return(MEMORY_ERROR);
    if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

    return (myNode[nodeIndex].H - myNode[nodeIndex].z);
 }


 /*!
  * \brief getTotalPotential
  * \param nodeIndex
  * \return total water potential (psi + z) [m]
  */
 double DLL_EXPORT __STDCALL getTotalPotential(long nodeIndex)
 {
	 if (myNode == nullptr) return(MEMORY_ERROR);
	 if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

	 return (myNode[nodeIndex].H);
 }


 /*!
  * \brief getWaterFlow
  * \param n
  * \param direction
  * \return maximum integrated flow in the requested direction [m^3]
  */
 double DLL_EXPORT __STDCALL getWaterFlow(long n, short direction)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((n < 0) || (n >= myStructure.nrNodes)) return INDEX_ERROR;

	double maxFlow = 0.0;

	switch (direction) {
        case UP:
            if (myNode[n].up.index != NOLINK)
            {
                return myNode[n].up.sumFlow;
            }
            else
            {
                return INDEX_ERROR;
            }

		case DOWN:
            if (myNode[n].down.index != NOLINK)
            {
                return myNode[n].down.sumFlow;
            }
            else
            {
                return INDEX_ERROR;
            }

		case LATERAL:
			// return maximum lateral flow
            for (short i = 0; i < myStructure.nrLateralLinks; i++)
                if (myNode[n].lateral[i].index != NOLINK)
                    if (fabs(myNode[n].lateral[i].sumFlow) > maxFlow)
                    {
                        maxFlow = myNode[n].lateral[i].sumFlow;
                    }

            return maxFlow;

        default:
            return INDEX_ERROR;
        }
 }


 /*!
  * \brief getSumLateralWaterFlow
  * \param n
  * \return integrated lateral flow over the time step [m^3]
  */
 double DLL_EXPORT __STDCALL getSumLateralWaterFlow(long n)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((n < 0) || (n >= myStructure.nrNodes)) return INDEX_ERROR;

    double sumLateralFlow = 0.0;
    for (short i = 0; i < myStructure.nrLateralLinks; i++)
    {
        if (myNode[n].lateral[i].index != NOLINK)
			sumLateralFlow += myNode[n].lateral[i].sumFlow;
    }
	return sumLateralFlow;
 }


 /*!
  * \brief getSumLateralWaterFlowIn
  * \param n
  * \return integrated lateral inflow over the time step [m^3]
  */
 double DLL_EXPORT __STDCALL getSumLateralWaterFlowIn(long n)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((n < 0) || (n >= myStructure.nrNodes)) return INDEX_ERROR;

    double sumLateralFlow = 0.0;
    for (short i = 0; i < myStructure.nrLateralLinks; i++)
        if (myNode[n].lateral[i].index != NOLINK)
            if (myNode[n].lateral[i].sumFlow > 0)
                sumLateralFlow += myNode[n].lateral[i].sumFlow;

    return sumLateralFlow;
 }


 /*!
  * \brief getSumLateralWaterFlowOut
  * \param n
  * \return integrated lateral outflow over the time step  [m^3]
  */
 double DLL_EXPORT __STDCALL getSumLateralWaterFlowOut(long n)
 {
    if (myNode == nullptr) return MEMORY_ERROR;
    if ((n < 0) || (n >= myStructure.nrNodes)) return INDEX_ERROR;

    double sumLateralFlow = 0.0;
    for (short i = 0; i < myStructure.nrLateralLinks; i++)
        if (myNode[n].lateral[i].index != NOLINK)
            if (myNode[n].lateral[i].sumFlow < 0)
                sumLateralFlow += myNode[n].lateral[i].sumFlow;

    return sumLateralFlow;
 }


 void DLL_EXPORT __STDCALL initializeBalance()
{
    InitializeBalanceWater();
    if (myStructure.computeHeat)
        initializeBalanceHeat();
}

 double DLL_EXPORT __STDCALL getWaterMBR()
 {
    return (balanceWholePeriod.waterMBR);
 }

 double DLL_EXPORT __STDCALL getHeatMBR()
  {
     return (balanceWholePeriod.heatMBR);
  }

 double DLL_EXPORT __STDCALL getHeatMBE()
  {
     return (balanceWholePeriod.heatMBE);
  }


 /*!
  * \brief getBoundaryWaterFlow
  * \param nodeIndex
  * \return integrated water flow from boundary over the time step [m^3]
  */
 double DLL_EXPORT __STDCALL getBoundaryWaterFlow(long nodeIndex)
 {
    if (myNode == nullptr)
        return MEMORY_ERROR;
    if (nodeIndex < 0 || nodeIndex >= myStructure.nrNodes)
        return INDEX_ERROR;
    if (myNode[nodeIndex].boundary == nullptr)
        return BOUNDARY_ERROR;

    return myNode[nodeIndex].boundary->sumBoundaryWaterFlow;
 }


 /*!
  * \brief getBoundaryWaterSumFlow
  * \param boundaryType
  * \return integrated water flow from all boundary over the time step  [m^3]
  */
 double DLL_EXPORT __STDCALL getBoundaryWaterSumFlow(int boundaryType)
 {
    double sumBoundaryFlow = 0.0;

    for (long n = 0; n < myStructure.nrNodes; n++)
        if (myNode[n].boundary != nullptr)
            if (myNode[n].boundary->type == boundaryType)
				sumBoundaryFlow += myNode[n].boundary->sumBoundaryWaterFlow;

    return sumBoundaryFlow;
 }


 /*!
  * \brief computePeriod
  * \param myPeriod
  * \return a period of time [s]
  */
 void DLL_EXPORT __STDCALL computePeriod(double myPeriod)
    {
        double sumTime = 0.0;

        balanceCurrentPeriod.sinkSourceWater = 0.;
        balanceCurrentPeriod.sinkSourceHeat = 0.;

        while (sumTime < myPeriod)
        {
            double ResidualTime = myPeriod - sumTime;
            sumTime += computeStep(ResidualTime);
        }

        if (myStructure.computeWater) updateBalanceWaterWholePeriod();
        if (myStructure.computeHeat) updateBalanceHeatWholePeriod();
    }


 /*!
 * \brief computeStep
 * \param maxTime
 * \return single step of time [s]
 */
double DLL_EXPORT __STDCALL computeStep(double maxTime)
{
    double dtWater, dtHeat;

    if (myStructure.computeHeat) initializeHeatFluxes(false, true);
    updateBoundary();

    if (myStructure.computeWater)
        computeWater(maxTime, &dtWater);
    else
        dtWater = MINVALUE(maxTime, myParameters.delta_t_max);

    dtHeat = dtWater;

    if (myStructure.computeHeat)
    {
        double dtHeatCurrent = dtHeat;

        saveWaterFluxes(dtHeatCurrent, dtWater);

        double dtHeatSum = 0;
        while (dtHeatSum < dtWater)
        {
            dtHeatCurrent = MINVALUE(dtHeat, dtWater - dtHeatSum);

            updateBoundaryHeat();

            if (HeatComputation(dtHeatCurrent, dtWater))
            {
                dtHeatSum += dtHeat;
            }
            else
            {
                restoreHeat();
                dtHeat = myParameters.current_delta_t;
            }
        }
    }

    return MINVALUE(dtWater, dtHeat);
}

/*!
 * \brief setTemperature
 * \param nodeIndex
 * \param myT [K]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setTemperature(long nodeIndex, double myT)
{
   //----------------------------------------------------------------------------------------------
   // Set current temperature of node
   //----- Input ----------------------------------------------------------------------------------
   // myT              [K] temperature
   //----------------------------------------------------------------------------------------------

   if (myNode == nullptr) return(MEMORY_ERROR);

   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

   if ((myT < 200) || (myT > 500)) return(PARAMETER_ERROR);

   if (! isHeatNode(nodeIndex)) return(MEMORY_ERROR);

   myNode[nodeIndex].extra->Heat->T = myT;
   myNode[nodeIndex].extra->Heat->oldT = myT;

   return(CRIT3D_OK);
}

/*!
 * \brief setFixedTemperature
 * \param nodeIndex
 * \param myT [K]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setFixedTemperature(long nodeIndex, double myT, double myDepth)
{
   if (myNode == nullptr) return(MEMORY_ERROR);
   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
   if (myNode[nodeIndex].boundary == nullptr) return(BOUNDARY_ERROR);
   if (myNode[nodeIndex].boundary->Heat == nullptr) return(BOUNDARY_ERROR);
   if (myNode[nodeIndex].boundary->type != BOUNDARY_PRESCRIBEDTOTALPOTENTIAL &&
           myNode[nodeIndex].boundary->type != BOUNDARY_FREEDRAINAGE) return(BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->fixedTemperatureDepth = myDepth;
   myNode[nodeIndex].boundary->Heat->fixedTemperature = myT;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryWindSpeed
 * \param nodeIndex
 * \param myWindSpeed [m s-1]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryWindSpeed(long nodeIndex, double myWindSpeed)
{
   if (myNode == nullptr) return(MEMORY_ERROR);
   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
   if ((myWindSpeed < 0) || (myWindSpeed > 1000)) return(PARAMETER_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->windSpeed = myWindSpeed;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryRoughness
 * \param nodeIndex
 * \param myRoughness [m]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryRoughness(long nodeIndex, double myRoughness)
{
   if (myNode == nullptr) return(MEMORY_ERROR);
   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
   if (myRoughness < 0) return(PARAMETER_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->roughnessHeight = myRoughness;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatSinkSource
 * \param nodeIndex
 * \param myHeatFlow [W]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatSinkSource(long nodeIndex, double myHeatFlow)
{
   if (myNode == nullptr)
       return(MEMORY_ERROR);

   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
       return(INDEX_ERROR);

   myNode[nodeIndex].extra->Heat->sinkSource = myHeatFlow;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryTemperature
 * \param nodeIndex
 * \param myTemperature [K]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryTemperature(long nodeIndex, double myTemperature)
{
   if (myNode == nullptr)
       return(MEMORY_ERROR);

   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
       return(INDEX_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->temperature = myTemperature;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryNetIrradiance
 * \param nodeIndex
 * \param myNetIrradiance [W m-2]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryNetIrradiance(long nodeIndex, double myNetIrradiance)
{
   if (myNode == nullptr)
       return(MEMORY_ERROR);

   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
       return(INDEX_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->netIrradiance = myNetIrradiance;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryRelativeHumidity
 * \param nodeIndex
 * \param myRelativeHumidity [%]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryRelativeHumidity(long nodeIndex, double myRelativeHumidity)
{
   if (myNode == nullptr)
       return(MEMORY_ERROR);

   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes))
       return(INDEX_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->relativeHumidity = myRelativeHumidity;

   return(CRIT3D_OK);
}

/*!
 * \brief setHeatBoundaryHeightWind
 * \param nodeIndex
 * \param myHeight [m]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryHeightWind(long nodeIndex, double myHeight)
{
   if (myNode == nullptr) return(MEMORY_ERROR);
   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->heightWind = myHeight;

   return(CRIT3D_OK);
}


/*!
 * \brief setHeatBoundaryHeightTemperature
 * \param nodeIndex
 * \param myHeight [m]
 * \return OK/ERROR
 */
int DLL_EXPORT __STDCALL setHeatBoundaryHeightTemperature(long nodeIndex, double myHeight)
{
   if (myNode == nullptr) return(MEMORY_ERROR);
   if ((nodeIndex < 0) || (nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);

   if (myNode[nodeIndex].boundary == nullptr || myNode[nodeIndex].boundary->Heat == nullptr)
       return (BOUNDARY_ERROR);

   myNode[nodeIndex].boundary->Heat->heightTemperature = myHeight;

   return(CRIT3D_OK);
}

/*!
 * \brief getTemperature
 * \param nodeIndex
 * \return temperature [K]
*/
double DLL_EXPORT __STDCALL getTemperature(long nodeIndex)
{
    if (myNode == nullptr) return(TOPOGRAPHY_ERROR);
    if ((nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
    if (! isHeatNode(nodeIndex)) return (MEMORY_ERROR);

    return (myNode[nodeIndex].extra->Heat->T);
}

/*!
 * \brief getHeatConductivity
 * \param nodeIndex
 * \return conductivity [W m-1 s-1]
 */
double DLL_EXPORT __STDCALL getHeatConductivity(long nodeIndex)
{
    if (myNode == nullptr) return(TOPOGRAPHY_ERROR);
    if ((nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
    if (! isHeatNode(nodeIndex)) return (MEMORY_ERROR);

   return SoilHeatConductivity(nodeIndex, myNode[nodeIndex].extra->Heat->T, myNode[nodeIndex].H - myNode[nodeIndex].z);
}

/*!
 * \brief getHeatFlux
 * \param nodeIndex
 * \param myDirection
 * \return instantaneous heat flux [W] or water flux (m3 -1)
*/
float DLL_EXPORT __STDCALL getHeatFlux(long nodeIndex, short myDirection, int fluxType)
{
    if (myNode == nullptr) return(TOPOGRAPHY_ERROR);
    if ((nodeIndex >= myStructure.nrNodes)) return(INDEX_ERROR);
    if (! isHeatNode(nodeIndex)) return (MEMORY_ERROR);

    float myMaxFlux = 0.;

    switch (myDirection)
    {
    case UP:
        return readHeatFlux(&(myNode[nodeIndex].up), fluxType);

    case DOWN:
        return readHeatFlux(&(myNode[nodeIndex].down), fluxType);

    case LATERAL:
        for (short i = 0; i < myStructure.nrLateralLinks; i++)
        {
            float myFlux = readHeatFlux(&(myNode[nodeIndex].lateral[i]), fluxType);
            if (myFlux != NODATA && myFlux > fabs(myMaxFlux))
                myMaxFlux = myFlux;
        }
        return myMaxFlux;

    default : return(INDEX_ERROR);
    }
}

/*!
 * \brief getBoundarySensibleFlux
 * \param nodeIndex
 * \return boundary sensible latent heat flux [W m-2]
*/
double DLL_EXPORT __STDCALL getBoundarySensibleFlux(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (INDEX_ERROR);

    // boundary sensible heat flow density
    return (myNode[nodeIndex].boundary->Heat->sensibleFlux);
}


/*!
 * \brief getBoundaryLatentFlux
 * \param nodeIndex
 * \return boundary latent heat flux [W m-2]
*/
double DLL_EXPORT __STDCALL getBoundaryLatentFlux(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat || ! myStructure.computeWater || ! myStructure.computeHeatVapor) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (INDEX_ERROR);

    // boundary latent heat flow density
    return (myNode[nodeIndex].boundary->Heat->latentFlux);
}

/*!
 * \brief getBoundaryAdvectiveFlux
 * \param nodeIndex
 * \return boundary advective heat flux [W m-2]
*/
double DLL_EXPORT __STDCALL getBoundaryAdvectiveFlux(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat || ! myStructure.computeWater || ! myStructure.computeHeatAdvection) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (BOUNDARY_ERROR);
    if (myNode[nodeIndex].boundary->Heat == nullptr) return (BOUNDARY_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (BOUNDARY_ERROR);

    // boundary advective heat flow density
    return (myNode[nodeIndex].boundary->Heat->advectiveHeatFlux);
}

/*!
 * \brief getBoundaryRadiativeFlux
 * \param nodeIndex
 * \return boundary radiative heat flux [W m-2]
*/
double DLL_EXPORT __STDCALL getBoundaryRadiativeFlux(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (INDEX_ERROR);

    // boundary net radiative heat flow density
    return (myNode[nodeIndex].boundary->Heat->radiativeFlux);
}

/*!
 * \brief getBoundaryAerodynamicConductance
 * \param nodeIndex
 * \return boundary aerodynamic conductance [m s-1]
*/
double DLL_EXPORT __STDCALL getBoundaryAerodynamicConductance(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (INDEX_ERROR);

    // boundary aerodynamic resistance
    return (myNode[nodeIndex].boundary->Heat->aerodynamicConductance);
}



/*!
 * \brief getBoundaryAerodynamicConductanceOpenWater
 * \param nodeIndex
 * \return boundary aerodynamic conductance [m s-1]
*/
/*
double DLL_EXPORT getBoundaryAerodynamicConductanceOpenWater(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if ((nodeIndex < 1) || (nodeIndex >= myStructure.nrNodes)) return (INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT) return (INDEX_ERROR);

    // boundary aerodynamic resistance
    return (myNode[nodeIndex].boundary->Heat->aerodynamicConductanceOpenwater);
}
*/

/*!
 * \brief getBoundarySoilConductance
 * \param nodeIndex
 * \return boundary soil conductance [m s-1]
*/
double DLL_EXPORT __STDCALL getBoundarySoilConductance(long nodeIndex)
{
    if (myNode == nullptr) return (TOPOGRAPHY_ERROR);
    if (nodeIndex >= myStructure.nrNodes) return (INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[nodeIndex].boundary == nullptr) return (INDEX_ERROR);
    if (myNode[nodeIndex].boundary->type != BOUNDARY_HEAT_SURFACE) return (INDEX_ERROR);

    // boundary soil conductance
    return (myNode[nodeIndex].boundary->Heat->soilConductance);
}

/*!
 * \brief getNodeVapor
 * \param nodeIndex
 * \return node vapor concentration [kg m-3]
*/
double DLL_EXPORT __STDCALL getNodeVapor(long i)
{
    if (myNode == nullptr) return(TOPOGRAPHY_ERROR);
    if (i >= myStructure.nrNodes) return(INDEX_ERROR);
    if (! myStructure.computeHeat || ! myStructure.computeWater || ! myStructure.computeHeatVapor) return (MISSING_DATA_ERROR);

    double h = myNode[i].H - myNode[i].z;
    double T = myNode[i].extra->Heat->T;

    return VaporFromPsiTemp(h, T);
}

/*!
 * \brief getHeat
 * \param nodeIndex
 * \return heat storage [J]
*/
double DLL_EXPORT __STDCALL getHeat(long i, double h)
{
    if (myNode == nullptr) return(TOPOGRAPHY_ERROR);
    if (i >= myStructure.nrNodes) return(INDEX_ERROR);
    if (! myStructure.computeHeat) return (MISSING_DATA_ERROR);
    if (myNode[i].extra->Heat == nullptr) return MISSING_DATA_ERROR;
    if (myNode[i].extra->Heat->T == NODATA) return MISSING_DATA_ERROR;

    double myHeat = SoilHeatCapacity(i, h, myNode[i].extra->Heat->T) * myNode[i].volume_area  * myNode[i].extra->Heat->T;

    if (myStructure.computeWater && myStructure.computeHeatVapor)
    {
        double thetaV = VaporThetaV(h, myNode[i].extra->Heat->T, i);
        myHeat += thetaV * latentHeatVaporization(myNode[i].extra->Heat->T - ZEROCELSIUS) * WATER_DENSITY * myNode[i].volume_area;
    }

    return (myHeat);
}

}
