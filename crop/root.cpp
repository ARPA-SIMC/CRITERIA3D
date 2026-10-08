/*!
    \file root.cpp

    \abstract
    root development functions

    \authors
    Antonio Volta       avolta@arpae.it
    Fausto Tomei        ftomei@arpae.it
    Gabriele Antolini   gantolini@arpae.it

    \copyright
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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
*/


#include <math.h>
#include <algorithm>

#include "commonConstants.h"
#include "basicMath.h"
#include "gammaFunction.h"
#include "root.h"
#include "crop.h"


Crit3DRoot::Crit3DRoot()
{
    this->clear();
}


void Crit3DRoot::clear()
{
    // parameters
    rootShape = CYLINDRICAL_DISTRIBUTION;
    growth = LOGISTIC;
    shapeDeformation = NODATA;
    degreeDaysRootGrowth = NODATA;
    rootDepthMin = NODATA;
    rootDepthMax = NODATA;

    // variables
    actualRootDepthMax = NODATA;
    firstRootLayer = NODATA;
    lastRootLayer = NODATA;
    currentRootLength = NODATA;
    rootDepth = NODATA;
    rootDensity.clear();
    rootsAdditionalCohesion = NODATA;
}


namespace root
{
    rootDistributionType getRootDistributionType(int rootShape)
    {
        switch (rootShape)
        {
            case (1):
                return CYLINDRICAL_DISTRIBUTION;
            case (4):
                return CARDIOID_DISTRIBUTION;
            case (5):
                return GAMMA_DISTRIBUTION;
            default:
                return CARDIOID_DISTRIBUTION;
         }
    }

    int getRootDistributionNumber(rootDistributionType rootShape)
    {
        switch (rootShape)
        {
            case (CYLINDRICAL_DISTRIBUTION):
                return 1;
            case (CARDIOID_DISTRIBUTION):
                return 4;
            case (GAMMA_DISTRIBUTION):
                return 5;
            default:
                // cardioid
                return 4;
         }
    }


    rootDistributionType getRootDistributionTypeFromString(const std::string &rootShape)
    {
        if (rootShape == "cylinder")
        {
            return CYLINDRICAL_DISTRIBUTION;
        }
        if (rootShape == "cardioid")
        {
            return CARDIOID_DISTRIBUTION;
        }
        if (rootShape == "gamma function")
        {
            return GAMMA_DISTRIBUTION;
        }

        // default
        return CARDIOID_DISTRIBUTION;
    }


    std::string getRootDistributionTypeString(rootDistributionType rootType)
    {
        switch (rootType)
        {
        case CYLINDRICAL_DISTRIBUTION:
            return "cylinder";
        case CARDIOID_DISTRIBUTION:
            return "cardioid";
        case GAMMA_DISTRIBUTION:
            return "gamma function";
        default:
            return "UNDEFINED";
        }
    }


    // [m]
    // this function computes the roots rate of development
    double getRootLengthDD(const Crit3DRoot &myRoot, double currentDD, double emergenceDD)
    {
        // in order to avoid numerical divergences when calculating density through cardioid and gamma function
        if (currentDD <= 1)
            return 0.;

        // growth phase ended
        double maxRootLength = myRoot.actualRootDepthMax - myRoot.rootDepthMin;
        if (currentDD > myRoot.degreeDaysRootGrowth)
            return maxRootLength;

        double currentRootLength = NODATA;
        if (myRoot.growth == LINEAR)
        {
            currentRootLength = maxRootLength * (currentDD / myRoot.degreeDaysRootGrowth);
        }
        else if (myRoot.growth == LOGISTIC)
        {
            double logMax, logMin,deformationFactor;
            double iniLog = log(9.);
            double filLog = log(1 / 0.99 - 1);
            double k = -(iniLog - filLog) / (emergenceDD - myRoot.degreeDaysRootGrowth);
            double b = -(filLog + k * myRoot.degreeDaysRootGrowth);

            logMax = (myRoot.actualRootDepthMax) / (1 + exp(-b - k * myRoot.degreeDaysRootGrowth));
            logMin = myRoot.actualRootDepthMax / (1 + exp(-b));
            deformationFactor = (logMax - logMin) / maxRootLength ;
            currentRootLength = 1.0 / deformationFactor * (myRoot.actualRootDepthMax / (1.0 + exp(-b - k * currentDD)) - logMin);
        }

        return currentRootLength;
    }


    int highestCommonFactor(int* vector, int vectorDim)
    {
        // highest common factor (hcf) amongst n integer numbers
        int num1, num2;
        int hcf = vector[0];
        for (int j=0; j<vectorDim-1; j++)
        {
            num1 = hcf;
            num2 = vector[j+1];

            for(int i=1; i<=num1 || i<=num2; ++i)
            {
                if(num1%i==0 && num2%i==0)   /* Checking whether i is a factor of both number */
                    hcf=i;
            }
        }
        return hcf;
    }


    int orderOfMagnitude(double number)
    {
        if (isEqual(number, 0.0))
            return 0;

        int order = floor(log10(fabs(number)));
        return order;
    }


    int getNrAtoms(const std::vector<soil::Crit1DLayer> &soilLayers, double &minThickness, std::vector<int> &atoms)
    {
        unsigned int nrLayers = unsigned(soilLayers.size());
        int multiplicationFactor = 1;

        minThickness = soilLayers[1].thickness;

        double tmp = minThickness * 1.001;
        if (tmp < 1)
            multiplicationFactor = int(pow(10.0, -orderOfMagnitude(tmp)));

        if (minThickness < 1)
        {
            minThickness = 1./multiplicationFactor;
        }

        int value;
        int counter = 0;
        for(unsigned int i=0; i < nrLayers; i++)
        {
           value = int(round(multiplicationFactor * soilLayers[i].thickness));
           atoms[i] = value;
           counter += value;
        }

        return counter;
    }


    /*!
     * \brief Compute root density distribution (cardioid)
     * \param shapeFactor: deformation factor [-]
     * \note author: Franco Zinoni
     * \return densityThinLayers [-] (vector)
     */
    bool cardioidDistribution(double shapeFactor, unsigned int nrLayersWithRoot,
                              unsigned int nrUpperLayersWithoutRoot, unsigned int totalLayers,
                              std::vector<double> &densityThinLayers)
    {
        // initialize
        densityThinLayers.assign(totalLayers, 0.0);

        // check
        if (nrLayersWithRoot == 0)
            return true;

        if (nrUpperLayersWithoutRoot + nrLayersWithRoot > totalLayers)
            return false;

        shapeFactor = std::clamp(shapeFactor, 1.0, 2.0);

        std::vector<double> lunette(nrLayersWithRoot);
        std::vector<double> lunetteDensity(nrLayersWithRoot * 2);

        double sinAlfa, cosAlfa, alfa;
        double halfPI = PI / 2.0;

        for (unsigned int i = 0; i < nrLayersWithRoot; ++i)
        {
            sinAlfa = 1.0 - double(i+1.0) / double(nrLayersWithRoot);
            double v = std::max(0.0, 1.0 - sinAlfa * sinAlfa);
            cosAlfa = std::max(std::sqrt(v), 0.0001);
            alfa = atan2(sinAlfa, cosAlfa);
            lunette[i] = (halfPI - alfa - sinAlfa * cosAlfa) / PI;
        }

        lunetteDensity[0] = lunette[0];
        lunetteDensity[2*nrLayersWithRoot - 1] = lunetteDensity[0];

        for (unsigned int i = 1; i < nrLayersWithRoot; ++i)
        {
            lunetteDensity[i] = lunette[i] - lunette[i-1];
            lunetteDensity[2*nrLayersWithRoot -i -1] = lunetteDensity[i];
        }

        // cardioid deformation
        double liMin = -std::log(0.2) / nrLayersWithRoot;
        double liMax = -std::log(0.05) / nrLayersWithRoot;

        double k = liMin + (liMax - liMin) * (shapeFactor-1);

        double rootDensitySum = 0.;
        for (unsigned int i = 0; i < 2 * nrLayersWithRoot; ++i)
        {
            lunetteDensity[i] *= std::exp(-k * (i + 0.5));
            rootDensitySum += lunetteDensity[i];
        }

        // normalize
        for (unsigned int i = 0; i < (2*nrLayersWithRoot); ++i)
            lunetteDensity[i] /= rootDensitySum;

        for (unsigned int i = 0; i < nrLayersWithRoot; ++i)
        {
            densityThinLayers[nrUpperLayersWithoutRoot+i] = lunetteDensity[2*i] + lunetteDensity[2*i+1];
        }

        return true;
    }


    bool cylindricalDistribution(double deformation, unsigned int nrLayersWithRoot,
                                 unsigned int nrUpperLayersWithoutRoot, unsigned int totalLayers,
                                 std::vector<double> &densityThinLayers)
    {
        // initialize
        densityThinLayers.assign(totalLayers, 0.0);

        // check
        if (nrLayersWithRoot == 0)
            return true;

        if (nrUpperLayersWithoutRoot + nrLayersWithRoot > totalLayers)
            return false;

        deformation = std::clamp(deformation, 0.0, 2.0);

        int nrLunette = 2*nrLayersWithRoot;
        double equalDensity = 1.0 / nrLunette;
        // initialize not deformed cylinder
        std::vector<double> cylinderDensity(nrLunette, equalDensity);

        // linear deformation
        double rootDensitySum = 0.0;
        double deltaDeformation = deformation - 1.0;

        for (int i = 0 ; i < nrLunette; i++)
        {
            deformation -= deltaDeformation / nrLayersWithRoot;
            cylinderDensity[i] *= deformation;
            rootDensitySum += cylinderDensity[i];
        }

        if (rootDensitySum <= EPSILON)
            return false;

        // normalize
        for (int i = 0; i < nrLunette; i++)
        {
           cylinderDensity[i] /= rootDensitySum;
        }

        for (unsigned int i = 0; i < nrLayersWithRoot ; i++)
        {
           densityThinLayers[nrUpperLayersWithoutRoot+i] = cylinderDensity[2*i] + cylinderDensity[2*i+1];
        }

        return true;
    }


    bool computeRootDensity(Crit3DCrop* myCrop, const std::vector<soil::Crit1DLayer> &soilLayers)
    {
        // check soil
        unsigned int nrLayers = unsigned(soilLayers.size());

        if (nrLayers == 0)
        {
            myCrop->roots.firstRootLayer = int(NODATA);
            myCrop->roots.lastRootLayer = int(NODATA);
            return false;
        }

        double soilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

        // Initialize
        myCrop->roots.rootDensity.assign(nrLayers, 0.0);

        if ((! myCrop->isLiving) || (myCrop->roots.currentRootLength <= 0 ))
            return true;

        if ((myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            || (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION))
        {
            double minimumThickness;
            std::vector<int> atoms;
            atoms.resize(nrLayers);
            const int nrAtoms = root::getNrAtoms(soilLayers, minimumThickness, atoms);

            const int numberOfTopUnrootedLayers = int(round(myCrop->roots.rootDepthMin / minimumThickness));
            int numberOfRootedLayers = int(round(std::min(myCrop->roots.currentRootLength, soilDepth) / minimumThickness));

            // roots are still too short
            if (numberOfRootedLayers == 0)
                return true;

            // check nr of thin layers
            if ((numberOfTopUnrootedLayers + numberOfRootedLayers) > nrAtoms)
            {
                numberOfRootedLayers = nrAtoms - numberOfTopUnrootedLayers;
            }

            // initialize thin layers density
            std::vector<double> densityThinLayers(nrAtoms, 0.0);

            if (myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            {
                if (! cardioidDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                          numberOfTopUnrootedLayers, nrAtoms, densityThinLayers))
                    return false;
            }
            else if (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION)
            {
                if (! cylindricalDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                             numberOfTopUnrootedLayers, nrAtoms, densityThinLayers))
                    return false;
            }

            int counter = 0;
            for (unsigned int layer = 0; layer < nrLayers; ++layer)
            {
                for (int j = 0; j < atoms[layer]; ++j)
                {
                    if (counter < nrAtoms)
                        myCrop->roots.rootDensity[layer] += densityThinLayers[counter];
                    counter++;
                }
            }
        }
        else if (myCrop->roots.rootShape == GAMMA_DISTRIBUTION)
        {
            double integralComplementary, kappa, theta, mode;

            double mean = myCrop->roots.currentRootLength * 0.5;
            int iterations=0;
            do{
                // TODO check (always kappa = mean / 0.4*mean = 2.5)
                mode = 0.6 * mean;
                theta = mean - mode;
                kappa = mean / theta;
                integralComplementary = incompleteGamma(kappa, 3 * myCrop->roots.currentRootLength/theta)
                                        - incompleteGamma(kappa, myCrop->roots.currentRootLength/theta);
                mean *= 0.99;
                iterations++;
            }
            while(integralComplementary > 0.01 && iterations < 1000);

            for (unsigned int i = 1 ; i < nrLayers; ++i)
            {
                const double b = std::max(0.0, soilLayers[i].depth
                                                   + soilLayers[i].thickness * 0.5
                                                   - myCrop->roots.rootDepthMin);       // right extreme

                if (b > 0 && b < myCrop->roots.currentRootLength)
                {
                    const double a = std::max(0.0, soilLayers[i].depth
                                                  - soilLayers[i].thickness * 0.5
                                                  - myCrop->roots.rootDepthMin);        // left extreme

                    // incompleteGamma is already normalized by gamma(kappa)
                    myCrop->roots.rootDensity[i] = incompleteGamma(kappa, b/theta)
                                                   - incompleteGamma(kappa, a/theta);
                }
                else
                {
                    myCrop->roots.rootDensity[i] = 0;
                }
            }
        }

        double rootDensitySum = 0.0;
        for (unsigned int i = 0 ; i < nrLayers; ++i)
        {
            myCrop->roots.rootDensity[i] *= soilLayers[i].soilFraction;
            rootDensitySum += myCrop->roots.rootDensity[i];
        }

        if (rootDensitySum <= EPSILON)
            return true;

        for (unsigned int i = 0 ; i < nrLayers ; ++i)
            myCrop->roots.rootDensity[i] /= rootDensitySum;

        myCrop->roots.firstRootLayer = int(NODATA);
        myCrop->roots.lastRootLayer = int(NODATA);
        for (unsigned int l = 0; l < nrLayers; ++l)
        {
            if (myCrop->roots.rootDensity[l] > EPSILON)
            {
                if (myCrop->roots.firstRootLayer == int(NODATA))
                    myCrop->roots.firstRootLayer = l;

                myCrop->roots.lastRootLayer = l;
            }
        }

        return true;
    }


    bool computeRootDensity3D(Crit3DCrop &myCrop, const soil::Crit3DSoil &currentSoil, unsigned int nrLayers,
                              const std::vector<double> &layerDepth, const std::vector<double> &layerThickness)
    {
        // check soil
        if (nrLayers <= 1)
        {
            myCrop.roots.firstRootLayer = int(NODATA);
            myCrop.roots.lastRootLayer = int(NODATA);
            return false;
        }

        // check vector size
        if (layerDepth.size() < nrLayers ||
            layerThickness.size() < nrLayers)
        {
            return false;
        }

        // initialize root density
        myCrop.roots.rootDensity.assign(nrLayers, 0.0);

        if (myCrop.roots.currentRootLength <= 0 )
            return true;

        // TODO Gamma distribution
        if (myCrop.roots.rootShape == GAMMA_DISTRIBUTION)
        {
            myCrop.roots.rootShape = CARDIOID_DISTRIBUTION;
        }

        const int nrAtoms = int(currentSoil.totalDepth * 100) + 1;
        const double oneCm = 0.01;                  // [m]
        const double minimumThickness = oneCm;      // [m]

        int numberOfRootedLayers, numberOfTopUnrootedLayers;
        numberOfTopUnrootedLayers = int(round(myCrop.roots.rootDepthMin / minimumThickness));
        numberOfRootedLayers = int(round(std::min(myCrop.roots.currentRootLength, currentSoil.totalDepth) / minimumThickness));

        // roots are still too short
        if (numberOfRootedLayers == 0)
            return true;

        // initialize thin layers density
        std::vector<double> densityThinLayers(nrAtoms, 0.0);

        // check nr of thin layers
        if ((numberOfTopUnrootedLayers + numberOfRootedLayers) > nrAtoms)
        {
            numberOfRootedLayers = nrAtoms - numberOfTopUnrootedLayers;
        }

        if (myCrop.roots.rootShape == CARDIOID_DISTRIBUTION)
        {
            if (! cardioidDistribution(myCrop.roots.shapeDeformation, numberOfRootedLayers,
                                      numberOfTopUnrootedLayers, nrAtoms, densityThinLayers))
                return false;
        }
        else if (myCrop.roots.rootShape == CYLINDRICAL_DISTRIBUTION)
        {
            if (! cylindricalDistribution(myCrop.roots.shapeDeformation, numberOfRootedLayers,
                                         numberOfTopUnrootedLayers, nrAtoms, densityThinLayers))
                return false;
        }

        double maxLayerDepth = layerDepth[nrLayers-1] + layerThickness[nrLayers-1] * 0.5;
        int atom = 0;
        double currentDepth = 0.0;                              // [m]
        double rootDensitySum = 0.0;                            // [-]
        while (currentDepth <= maxLayerDepth && atom < nrAtoms)
        {
            for (unsigned int l = 0; l < nrLayers; l++)
            {
                double upperDepth = layerDepth[l] - layerThickness[l] * 0.5;
                double lowerDepth = layerDepth[l] + layerThickness[l] * 0.5;
                if (currentDepth >= upperDepth && currentDepth <= lowerDepth)
                {
                    myCrop.roots.rootDensity[l] += densityThinLayers[atom];
                    rootDensitySum += densityThinLayers[atom];
                    break;
                }
            }

            atom++;
            currentDepth = double(atom) * oneCm;                // [m]
        }

        if (rootDensitySum <= EPSILON)
            return true;

        double rootDensitySumSubset = 0.;
        for (unsigned int l=0 ; l < nrLayers; l++)
        {
            int horIndex = currentSoil.getHorizonIndex(layerDepth[l]);
            if (horIndex != int(NODATA))
            {
                myCrop.roots.rootDensity[l] *= currentSoil.horizon[horIndex].getSoilFraction();
                rootDensitySumSubset += myCrop.roots.rootDensity[l];
            }
        }

        // normalize root density
        if (rootDensitySumSubset > EPSILON &&
            std::abs(rootDensitySumSubset - rootDensitySum) > EPSILON)
        {
            double ratio = rootDensitySum / rootDensitySumSubset;

            for (unsigned int l = 0; l < nrLayers; ++l)
            {
                myCrop.roots.rootDensity[l] *= ratio;
            }
        }

        // first and last root layers
        myCrop.roots.firstRootLayer = int(NODATA);
        myCrop.roots.lastRootLayer = int(NODATA);

        for (unsigned int l = 0; l < nrLayers; ++l)
        {
            if (myCrop.roots.rootDensity[l] > EPSILON)
            {
                if (myCrop.roots.firstRootLayer == int(NODATA))
                {
                    myCrop.roots.firstRootLayer = static_cast<int>(l);
                }

                myCrop.roots.lastRootLayer = static_cast<int>(l);
            }
        }

        return true;
    }
}

