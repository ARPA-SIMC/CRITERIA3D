#ifndef CROP_H
#define CROP_H

    #ifndef _STRING_
        #include <string>
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef ROOT_H
        #include "root.h"
    #endif

    enum speciesType {HERBACEOUS_ANNUAL, HERBACEOUS_PERENNIAL, HORTICULTURAL, GRASS, FALLOW, FRUIT_TREE};
    const int numSpeciesType = 6;

    /*!
     * \brief The Crit3DCrop class
     */
    class Crit3DCrop
    {
        public:

        std::string idCrop;
        speciesType type;

        Crit3DRoot roots;

        /*!
         * crop cycle
         */
        int sowingDoy;
        int currentSowingDoy;
        int doyStartSenescence;
        int plantCycle;
        double LAImin, LAImax, LAIgrass;
        double LAIcurve_a, LAIcurve_b;
        double thermalThreshold;
        double upperThermalThreshold;
        double degreeDaysIncrease, degreeDaysDecrease, degreeDaysEmergence;

        /*!
         * water need
         */
        double kcMax;                               /*!< [-] */
        double psiLeaf;                             /*!< [cm] */
        double stressTolerance;                     /*!< [-] */
        double fRAW;                                /*!< [-] fraction of Readily Available Water */

        /*!
         * irrigation
         */
        int irrigationShift;
        double irrigationVolume;
        int degreeDaysStartIrrigation, degreeDaysEndIrrigation;
        int doyStartIrrigation, doyEndIrrigation;
        double maxSurfacePuddle;

        /*!
         * variables
         */
        double degreeDays;
        bool isLiving;
        bool isEmerged;
        double LAI;
        double LAIstartSenescence;
        int daysSinceIrrigation;

        Crit3DCrop();

        bool isWaterSurplusResistant();
        int getDaysFromTypicalSowing(int myDoy);
        int getDaysFromCurrentSowing(int myDoy);
        bool isInsideTypicalCycle(int myDoy);
        bool isPluriannual();

        void initialize(double latitude, unsigned int nrLayers, double totalSoilDepth, int currentDoy);
        bool needReset(Crit3DDate myDate, double latitude, double waterTableDepth);
        void resetCrop(unsigned int nrLayers);
        bool updateLAI(double latitude, unsigned int nrLayers, int myDoy);
        bool dailyUpdate(const Crit3DDate &myDate, double latitude, unsigned int nrLayers, double totalDepth,
                        double tmin, double tmax, double waterTableDepth, std::string* myError);

        double getSurfaceCoverFraction();
        double getMaxEvaporation(double ET0);
        double getMaxTranspiration(double ET0);

        double getCropWaterDeficit(const std::vector<soil::Crit3DLayer> &layers);
        double getIrrigationDemand(int doy, double currentPrec, double nextPrec,
                                   double maxTranpiration, const std::vector<soil::Crit3DLayer>& layers);

        double computeTranspiration(double maxTranspiration, const std::vector<soil::Crit3DLayer>& layers,
                                    std::vector<double>* layerTranspiration, bool returnWaterStress);
    };


    speciesType getCropType(std::string cropType);
    std::string getCropTypeString(speciesType cropType);

    double computeDegreeDays(double myTmin, double myTmax, double myLowerThreshold, double myUpperThreshold);


#endif // CROP_H
