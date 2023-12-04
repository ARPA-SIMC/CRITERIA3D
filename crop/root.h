#ifndef ROOT_H
#define ROOT_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #include <vector>

    class Crit3DCrop;

    enum rootDistributionType {CYLINDRICAL_DISTRIBUTION, CARDIOID_DISTRIBUTION, GAMMA_DISTRIBUTION};
    const int nrRootDistributionType = 3;

    enum rootGrowthType {LINEAR, EXPONENTIAL, LOGISTIC};

    /*!
     * \brief The Crit3DRoot class
     */
    class Crit3DRoot
    {
    public:
        rootDistributionType rootShape;
        rootGrowthType growth;

        /*! parameters */
        int degreeDaysRootGrowth;           /*!< [°D]  */
        double rootDepthMin;                /*!< [m]   */
        double rootDepthMax;                /*!< [m]   */
        double shapeDeformation;            /*!< [-]   */

        /*! variables */
        double actualRootDepthMax;          /*!< [m]  it takes into account soil depth */
        double currentRootLength;           /*!< [m]  */
        int firstRootLayer;                 /*!< [-]  */
        int lastRootLayer;                  /*!< [-]  */
        std::vector<double> rootDensity;    /*!< [-]  */
        double rootsAdditionalCohesion;     /*!< [kPa] Cr = roots reinforcement (RR) derived from a model */

        /*! state variables */
        double rootDepth;                   /*!<  [m]  current root depth */

        Crit3DRoot();

        void clear();

    };

    namespace root
    {
        rootDistributionType getRootDistributionType(int rootShape);
        int getRootDistributionNumber(rootDistributionType rootShape);
        rootDistributionType getRootDistributionTypeFromString(const std::string &rootShape);
        std::string getRootDistributionTypeString(rootDistributionType rootType);

        double getRootLengthDD(const Crit3DRoot &myRoot, double currentDD, double emergenceDD);
        bool computeRootDensity(Crit3DCrop* myCrop, const std::vector<soil::Crit3DLayer> &soilLayers);

        int getNrAtoms(const std::vector<soil::Crit3DLayer> &soilLayers, double &minThickness, std::vector<int> &atoms);
    }

#endif // ROOT_H
