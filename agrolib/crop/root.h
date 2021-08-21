#ifndef ROOT_H
#define ROOT_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #include <vector>

    class Crit3DCrop;

    enum rootDistributionType {CYLINDRICAL_DISTRIBUTION, CARDIOID_DISTRIBUTION, GAMMA_DISTRIBUTION};
    const int numRootDistributionType = 3;
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
        double degreeDaysRootGrowth;        /*!< [°D]  */
        double rootDepthMin;                /*!< [m]   */
        double rootDepthMax;                /*!< [m]   */
        double shapeDeformation;            /*!< [-]   */

        /*! variables */
        double actualRootDepthMax;          /*!< [m]  it takes into account soilDepth */
        double rootLength;                  /*!< [m]  */
        int firstRootLayer;                 /*!< [-]  */
        int lastRootLayer;                  /*!< [-]  */
        std::vector<double> rootDensity;    /*!< [-]  */

        /*! state variables */
        double rootDepth;                   /*!<  [m]  current root depth */

        Crit3DRoot();

        void clear();

    };

    namespace root
    {
        int getNrAtoms(const std::vector<soil::Crit3DLayer> &soilLayers, double rootDepthMin, double* minThickness, int* atoms);
        double getRootLengthDD(Crit3DRoot* myRoot, double currentDD, double emergenceDD);
        rootDistributionType getRootDistributionType(int rootShape);
        rootDistributionType getRootDistributionTypeFromString(std::string rootShape);
        std::string getRootDistributionTypeString(rootDistributionType rootType);

        double computeRootLength(Crit3DCrop* myCrop, double currentDD, double waterTableDepth);
        double computeRootDepth(Crit3DCrop* myCrop, double currentDD, double waterTableDepth);
        bool computeRootDensity(Crit3DCrop* myCrop, const std::vector<soil::Crit3DLayer> &soilLayers);
    }

#endif // ROOT_H
