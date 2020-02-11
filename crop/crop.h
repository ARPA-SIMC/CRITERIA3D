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
        bool needReset(Crit3DDate myDate, float latitude, float waterTableDepth);
        void resetCrop(int nrLayers);
    };


    speciesType getCropType(std::string cropType);
    std::string getCropTypeString(speciesType cropType);

    double computeDegreeDays(double myTmin, double myTmax, double myLowerThreshold, double myUpperThreshold);


#endif // CROP_H
