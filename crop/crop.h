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

    enum speciesType {HERBACEOUS_ANNUAL, HERBACEOUS_PERENNIAL, HORTICULTURAL, GRASS, TREE, FALLOW, FALLOW_ANNUAL};
    #define NR_CROP_SPECIES 7

    /*!
     * \brief The Crit3DCrop class
     */
    class Crit3DCrop
    {
        public:

        std::string idCrop;
        std::string name;
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
        int psiLeaf;                                /*!< [cm] */
        double stressTolerance;                     /*!< [-] */
        double fRAW;                                /*!< [-] fraction of Readily Available Water */

        /*!
         * irrigation
         */
        int irrigationShift;
        double irrigationVolume;                    /*!< [mm] */
        int degreeDaysStartIrrigation, degreeDaysEndIrrigation;
        int doyStartIrrigation, doyEndIrrigation;
        double maxSurfacePuddle;                    /*!< [mm] */

        /*!
         * variables
         */
        double degreeDays;
        bool isLiving;
        bool isEmerged;
        double LAI;
        double LAIpreviousDay;
        double LAIstartSenescence;
        int daysSinceIrrigation;
        std::vector<double> layerTranspiration;

        Crit3DCrop();

        void clear();

        int getDaysFromTypicalSowing(int myDoy) const;
        int getDaysFromCurrentSowing(int myDoy) const;
        bool isInsideTypicalCycle(int myDoy) const;

        bool isWaterSurplusResistant() const;
        bool isSowingCrop() const;
        bool isRootStatic() const;

        double getDailyDegreeIncrease(double tmin, double tmax, int doy);

        void initialize(double latitude, unsigned int nrLayers, double totalSoilDepth, int currentDoy);
        bool needReset(Crit3DDate myDate, double latitude, double waterTableDepth);
        void resetCrop(unsigned int nrLayers);

        bool updateLAI(double latitude, unsigned int nrLayers, int currentDoy);
        void updateRootDepth(double currentDD, double waterTableDepth);
        double computeRootLength(double currentDD, double waterTableDepth);

        void computeRootLength3D(double currentDD, double totalSoilDepth);

        double computeSimpleLAI(double myDegreeDays, double latitude, int currentDoy);

        bool dailyUpdate(const Crit3DDate &myDate, double latitude, const std::vector<soil::Crit3DLayer> &soilLayers,
                         double tmin, double tmax, double waterTableDepth, std::string &myError);
        bool restore(const Crit3DDate &myDate, double latitude, const std::vector<soil::Crit3DLayer> &soilLayers,
                     double currentWaterTable, std::string &myError);

        double getCoveredSurfaceFraction();
        double getMaxEvaporation(double ET0);
        double getMaxTranspiration(double ET0);
        double getSurfaceWaterPonding() const;

        double getCropWaterDeficit(const std::vector<soil::Crit3DLayer> & soilLayers);

        double computeTranspiration(double maxTranspiration, const std::vector<soil::Crit3DLayer>& soilLayers, double& waterStress);
    };


    speciesType getCropType(std::string cropType);
    std::string getCropTypeString(speciesType cropType);


#endif // CROP_H
