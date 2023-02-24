#ifndef CARBON_NITROGEN_MODEL_H
#define CARBON_NITROGEN_MODEL_H

    #include "crit3dDate.h"
    #include "carbonNitrogen.h"
    #include "criteria1DCase.h"

    class Crit1DCarbonNitrogenProfile
    {
    public:
        Crit1DCarbonNitrogenProfile();

        Crit3DCarbonNitrogenSettings carbonNitrogenParameter;
        // values corrected for Temperature and RH

        TactualRate actualRate;
        // fix variables --------------------------------------------------------------------------------------------
        double ratio_CN_humus;               // [] ratio C/N pool humus
        double ratio_CN_biomass;             // [] ratio C/N pool biomass

        double litterIniC;                   //[kg ha-1] initial litter carbon
        double litterIniN;                   //[kg ha-1] initial litter nitrogen
        double litterIniDepth ;               //[cm] initial litter depth

        // flags -------------------------------------------------------------------------------------------------
        TFlag flag;
        // daily values---------------------------------------------------------------------------------
        // Nitrogen in soil
        // contents

        // uptake
        Crit3DDate date_N_endCrop;
        Crit3DDate date_N_plough;

        //double N_uptakable;              //[g m-2] assorbimento massimo della coltura per ciclo colturale
    private:
        double maxRate_LAI_Ndemand;      //[g m-2 d-1 LAI-1] maximum demand for unit LAI increment

            //carbon
        // content
        TNitrogenTotalProfile nitrogenTotalProfile;
        TCarbonTotalProfile carbonTotalProfile;

    public:

        void N_main(double precGG, Crit1DCase &myCase, Crit3DDate &myDate);
        void N_InitializeVariables(Crit1DCase &myCase);

    private:
        void N_Initialize();
        void humus_Initialize(Crit1DCase &myCase);
        void litter_Initialize(Crit1DCase &myCase);
        void N_initializeCrop(bool noReset,Crit1DCase &myCase);

        double updateTotalOfPartitioned(double mySoluteAds,double mySoluteSol);
        void partitioning(Crit1DCase &myCase);
        void chemicalTransformations(Crit1DCase &myCase);
        void N_Fertilization(Crit1DCase &myCase, Crit3DFertilizerProperties fertilizerProperties);

        //void ApriTabellaUsciteAzoto(tbname_azoto As String);
        void N_Output();
        double computeWaterCorrectionFactor(int l,Crit1DCase &myCase);
        double computeTemperatureCorrectionFactor(bool flag, double layerSoilTemperature, double baseTemperature);
        void computeLayerRates(unsigned l,Crit1DCase &myCase);
        void N_Uptake(Crit1DCase &myCase);
        void N_SurfaceRunoff(Crit1DCase &myCase);
        void N_SubSurfaceRunoff();
        void N_Uptake_Potential(Crit1DCase &myCase);
        void N_Uptake_Max();
        void N_Reset();
        double findPistonDepth(Crit1DCase &myCase);
        void soluteFluxesPiston(double* mySolute, double PistonDepth,double* leached);
        void soluteFluxesPiston_old(double* mySolute, double* leached, double* CoeffPiston);
        void soluteFluxes(std::vector<double> &mySolute, bool flagRisalita, double pistonDepth, double* leached, Crit1DCase &myCase);
        void leachingWaterTable(std::vector<double> &mySolute, double* leached, Crit1DCase &myCase);
        void NH4_Balance(Crit1DCase &myCase);
        void NO3_Balance(Crit1DCase &myCase);

        void N_harvest(Crit1DCase &myCase);
        void updateNCrop(Crit3DCrop crop);
        void N_plough(Crit1DCase &myCase);
        void NFromCropSenescence(double myDays, double coeffB, Crit1DCase &myCase);

    };

#endif // CARBON_NITROGEN_MODEL_H
