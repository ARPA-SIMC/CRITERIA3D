#ifndef SOILFLUXES3D
#define SOILFLUXES3D

    // Uncomment to compile as win32 dll
    // #define BUILD_DLL 1

    #ifdef BUILD_DLL
        #define DLL_EXPORT __declspec(dllexport)
        #define __EXTERN extern "C"
	    #define __STDCALL __stdcall
    #else
        #define DLL_EXPORT
        #define __EXTERN
        #define __STDCALL
    #endif
	
    namespace soilFluxes3D {

    // TEST
    __EXTERN int DLL_EXPORT __STDCALL test();

    // INITIALIZATION
    __EXTERN void DLL_EXPORT __STDCALL cleanMemory();
    __EXTERN int DLL_EXPORT __STDCALL initializeFluxes(long nrNodes, int nrLayers, int nrLateralLinks, bool isComputeWater, bool isComputeHeat, bool isComputeSolutes);
    __EXTERN void DLL_EXPORT __STDCALL initializeHeat(short saveHeatFluxes_, bool computeAdvectiveHeat, bool computeLatentHeat);

    __EXTERN int DLL_EXPORT __STDCALL setNumericalParameters(float minDeltaT, float maxDeltaT,
                              int maxIterationNumber, int maxApproximationsNumber,
                              int errorMagnitude, float MBRMagnitude);

    __EXTERN int DLL_EXPORT __STDCALL setThreads(int nrThreads);

    // TOPOLOGY
    __EXTERN int DLL_EXPORT __STDCALL setNode(long myIndex, float x, float y, double z, double volume_or_area,
                                        bool isSurface, bool isBoundary, int boundaryType, float slope, float boundaryArea);

    __EXTERN int DLL_EXPORT __STDCALL setNodeLink(long nodeIndex, long linkIndex, short direction, float S0);

	__EXTERN int DLL_EXPORT __STDCALL setCulvert(long myIndex, double roughness, double slope, double width, double height);

    // SOIL
    __EXTERN int DLL_EXPORT __STDCALL setSoilProperties(int nrSoil, int nrHorizon, double VG_alpha,
                                        double VG_n, double VG_m, double VG_he,
                                        double ThetaR, double ThetaS, double Ksat, double L,
                                        double organicMatter, double clay);

    __EXTERN int DLL_EXPORT __STDCALL setNodeSoil(long nodeIndex, int soilIndex, int horizonIndex);

    // SURFACE
    __EXTERN int DLL_EXPORT __STDCALL setSurfaceProperties(int surfaceIndex, double Roughness);
    __EXTERN int DLL_EXPORT __STDCALL setNodeSurface(long nodeIndex, int surfaceIndex);
    __EXTERN int DLL_EXPORT __STDCALL setNodePond(long nodeIndex, float pond);

    // WATER
    __EXTERN int DLL_EXPORT __STDCALL setHydraulicProperties(int waterRetentionCurve, int conductivityMeanType, float conductivityHorizVertRatio);
    __EXTERN int DLL_EXPORT __STDCALL setWaterContent(long index, double myWaterContent);
    __EXTERN int DLL_EXPORT __STDCALL setDegreeOfSaturation(long nodeIndex, double degreeOfSaturation);
    __EXTERN int DLL_EXPORT __STDCALL setMatricPotential(long index, double psi);
    __EXTERN int DLL_EXPORT __STDCALL setTotalPotential(long index, double totalPotential);
    __EXTERN int DLL_EXPORT __STDCALL setPrescribedTotalPotential(long index, double prescribedTotalPotential);
    __EXTERN int DLL_EXPORT __STDCALL setWaterSinkSource(long index, double sinkSource);

    __EXTERN double DLL_EXPORT __STDCALL getWaterContent(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getMaximumWaterContent(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getAvailableWaterContent(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getWaterDeficit(long index, double fieldCapacity);
    __EXTERN double DLL_EXPORT __STDCALL getTotalWaterContent();
    __EXTERN double DLL_EXPORT __STDCALL getDegreeOfSaturation(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryWaterFlow(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryWaterSumFlow(int boundaryType);
    __EXTERN double DLL_EXPORT __STDCALL getMatricPotential(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getTotalPotential(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getWaterMBR();
    __EXTERN double DLL_EXPORT __STDCALL getWaterConductivity(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getWaterFlow(long nodeIndex, short direction);
    __EXTERN double DLL_EXPORT __STDCALL getSumLateralWaterFlow(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getSumLateralWaterFlowIn(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getSumLateralWaterFlowOut(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getWaterStorage();
    __EXTERN float DLL_EXPORT __STDCALL getPond(long nodeIndex);

    // HEAT
    __EXTERN int DLL_EXPORT __STDCALL setHeatSinkSource(long nodeIndex, double myHeatFlow);
    __EXTERN int DLL_EXPORT __STDCALL setTemperature(long nodeIndex, double myT);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryHeightWind(long nodeIndex, double myHeight);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryHeightTemperature(long nodeIndex, double myHeight);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryTemperature(long nodeIndex, double myTemperature);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryRelativeHumidity(long nodeIndex, double myRelativeHumidity);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryRoughness(long nodeIndex, double myRoughness);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryWindSpeed(long nodeIndex, double myWindSpeed);
    __EXTERN int DLL_EXPORT __STDCALL setHeatBoundaryNetIrradiance(long nodeIndex, double myNetIrradiance);
    __EXTERN int DLL_EXPORT __STDCALL setFixedTemperature(long nodeIndex, double myT, double myDepth);

    __EXTERN double DLL_EXPORT __STDCALL getTemperature(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getHeatConductivity(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getHeat(long nodeIndex, double h);
    __EXTERN double DLL_EXPORT __STDCALL getNodeVapor(long nodeIndex);
    __EXTERN float DLL_EXPORT __STDCALL getHeatFlux(long nodeIndex, short myDirection, int fluxType);
    __EXTERN double DLL_EXPORT __STDCALL getBoundarySensibleFlux(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryAdvectiveFlux(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryLatentFlux(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryRadiativeFlux(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundaryAerodynamicConductance(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getBoundarySoilConductance(long nodeIndex);
    __EXTERN double DLL_EXPORT __STDCALL getHeatMBR();
    __EXTERN double DLL_EXPORT __STDCALL getHeatMBE();

    // SOLUTES
    // ...

    // COMPUTATION
    __EXTERN void DLL_EXPORT __STDCALL initializeBalance();
    __EXTERN void DLL_EXPORT __STDCALL computePeriod(double myPeriod);
	__EXTERN double DLL_EXPORT __STDCALL computeStep(double maxTime);

}

#endif
