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
        int flagSOM;                        // 1: computes SO; 0: SO set at the default value
        int flagLocalOS;                    // 1: Initializes the profile of SO without keeping that of soil
        bool flagWaterTableWashing;         // if true: the solute is completely leached in groundwater
        bool flagWaterTableUpward;          // if true: capillary rise is allowed

        // daily values---------------------------------------------------------------------------------
        // Nitrogen in soil
        // contents
        double N_humusGG;                //[g m-2] Nitrogen within humus
        double N_litterGG;               //[g m-2] Nitrogen within litter
        double N_NH4_adsorbedGG;         //[g m-2] adsorbed Ammonium in the current day
        double N_NH4_adsorbedBeforeGG;   //[g m-2] adsorbed Ammonium in the previous day
        double profileNO3;              //[g m-2] N-NO3 in the whole profile
        double profileNH4;              //[g m-2] N-NH4 in the whole profile
        double balanceFinalNO3;            //[g m-2] N-NO3: budget error
        double balanceFinalNH4;            //[g m-2] N-NH4: budget error
        //fluxes
        double precN_NO3GG;              //[g m-2] NO3 from rainfall
        double precN_NH4GG;              //[g m-2] NH4 from rainfall
        double N_NO3_fertGG;             //[g m-2] NO3 from fertilization
        double N_NH4_fertGG;             //[g m-2] NH4 from fertilization
        double N_min_litterGG;           //[g m-2] mineralized Nitrogen from litter

    private:
        double N_imm_l_NH4GG;            //[g m-2] NH4 immobilized in litter
        double N_imm_l_NO3GG;            //[g m-2] NO3 immobilized in litter
    public:
        double N_min_humusGG;            //[g m-2] mineralized Nitrogen from humus
        double N_litter_humusGG;         //[g m-2] Nitrogen from litter to humus
        double N_NH4_volGG;              //[g m-2] Volatilized NH4 in the whole profile
        double N_nitrifGG;               //[g m-2] Nitrogen from NH4 to NO3
        double N_urea_hydrGG;            //[g m-2] Hydrolyzed urea urea to NH4
        double flux_NO3GG;               //[g m-2] NO3 leaching flux
        double flux_NH4GG;               //[g m-2] NH4 leaching flux
        double N_NO3_runoff0GG;          //[g m-2] NO3 lost through surface run off
        double N_NH4_runoff0GG;          //[g m-2] NH4 lost through surface run off
        double N_NO3_runoffGG;           //[g m-2] NO3 lost through subsurface run off
        double N_NH4_runoffGG;           //[g m-2] NH4 lost through subsurface run off
        // uptake
        Crit3DDate date_N_endCrop;
        Crit3DDate date_N_plough;

        double N_uptakable;              //[g m-2] assorbimento massimo della coltura per ciclo colturale
    private:
        double maxRate_LAI_Ndemand;      //[g m-2 d-1 LAI-1] maximum demand for unit LAI increment

    public:
        double N_cropToHarvest;          //[g m-2] Nitrogen absorbed in harvest
        double N_cropToResidues;         //[g m-2] Nitrogen absorbed in crop residues
        double N_roots;                  //[g m-2] Nitrogen absorbed in roots
    private:
        double N_ratioHarvested;         //[] ratio of harvested crop
        double N_ratioResidues;          //[] ratio of residues not harvested left above the soil
        double N_ratioRoots;             //[] ratio of living roots left at harvest

    public:
        double N_potentialDemandCum;     //[g m-2] cumulated potential Nitrogen at current date
        double N_dailyDemand;            //[g m-2] potential Nitrogen at current date
        double N_dailyDemandMaxCover;    //[g m-2] potential Nitrogen at max cover day (LAI_MAX)
        double N_uptakeMax;              //[g m-2] Max Nitrogen uptake
        double N_uptakeDeficit;          //[g m-2] Nitrogen deficit: not absorbed Nitrogen with respect to Nitrogen demand

        int N_deficit_max_days;          //[d] nr days with available deficit
        double N_NH4_uptakeGG;           //[g m-2] NH4 absorbed by the crop
        double N_NO3_uptakeGG;           //[g m-2] NO3 absorbed by the crop
        double N_denitrGG;               //[g m-2] Lost Nitrogen by denitrification
            //carbon
        // content
        double C_humusGG;                //[g m-2] C in humus
        double C_litterGG;               //[g m-2] C in litter
        // flux
        double C_litter_humusGG;         //[g m-2] C from litter to humus
        double C_litter_litterGG;        //[g m-2] C recycled within litter
        double C_min_humusGG;            //[g m-2] C lost as CO2 by humus mineralization
        double C_min_litterGG;           //[g m-2] C lost as CO2 by litter mineralization

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
