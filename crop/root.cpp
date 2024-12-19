/*!
    \file root.cpp

    \abstract
    root development functions

    \authors
    Antonio Volta       avolta@arpae.it
    Fausto Tomei        ftomei@arpae.it
    Gabriele Antolini   gantolini@arpe.it

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

#include "commonConstants.h"
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
        }

        return "Undefined root type";
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


    int checkTheOrderOfMagnitude(double number, int &order)
    {
        if (number<1)
        {
            number *= 10;
            order--;
            checkTheOrderOfMagnitude(number, order);
        }
        else if (number >= 10)
        {
            number /=10;
            order++;
            checkTheOrderOfMagnitude(number, order);
        }
        return 0;
    }


    int orderOfMagnitude(double number)
    {
        int order = 0;
        number = fabs(number);
        checkTheOrderOfMagnitude(number, order);
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
     * \return densityThinLayers [-] (array)
     */
    void cardioidDistribution(double shapeFactor, unsigned int nrLayersWithRoot,
                              unsigned int nrUpperLayersWithoutRoot, unsigned int totalLayers,
                              std::vector<double> &densityThinLayers)
    {
        unsigned int i;
        std::vector<double> lunette, lunetteDensity;
        lunette.resize(nrLayersWithRoot);
        lunetteDensity.resize(nrLayersWithRoot*2);

        double sinAlfa, cosAlfa, alfa;
        for (i = 0; i < nrLayersWithRoot; i++)
        {
            sinAlfa = 1 - double(i+1) / double(nrLayersWithRoot);
            cosAlfa = MAXVALUE(sqrt(1 - pow(sinAlfa,2)), 0.0001);
            alfa = atan(sinAlfa/cosAlfa);
            lunette[i] = ((PI/2) - alfa - sinAlfa*cosAlfa) / PI;
        }

        lunetteDensity[0] = lunette[0];
        lunetteDensity[2*nrLayersWithRoot - 1] = lunetteDensity[0];
        for (i = 1; i < nrLayersWithRoot; i++)
        {
            lunetteDensity[i] = lunette[i] - lunette[i-1];
            lunetteDensity[2*nrLayersWithRoot -i -1] = lunetteDensity[i];
        }

        // cardioid deformation
        double LiMin,Limax,k,rootDensitySum ;
        LiMin = -log(0.2) / nrLayersWithRoot;
        Limax = -log(0.05) / nrLayersWithRoot;

        // TODO verify
        k = LiMin + (Limax - LiMin) * (shapeFactor-1);

        rootDensitySum = 0 ;
        for (i = 0; i < (2*nrLayersWithRoot); i++)
        {
            lunetteDensity[i] *= exp(-k*(i+0.5));
            rootDensitySum += lunetteDensity[i];
        }
        for (i = 0; i < (2*nrLayersWithRoot); i++)
        {
            lunetteDensity[i] /= rootDensitySum;
        }
        for  (i = 0; i < totalLayers; i++)
        {
            densityThinLayers[i] = 0;
        }
        for (i = 0; i < nrLayersWithRoot; i++)
        {
            densityThinLayers[nrUpperLayersWithoutRoot+i] = lunetteDensity[2*i] + lunetteDensity[2*i+1];
        }

        lunette.clear();
        lunetteDensity.clear();
    }


    void cylindricalDistribution(double deformation, unsigned int nrLayersWithRoot,
                                 unsigned int nrUpperLayersWithoutRoot, unsigned int totalLayers,
                                 std::vector<double> &densityThinLayers)
    {
       unsigned int i;
       std::vector<double> cylinderDensity;
       cylinderDensity.resize(nrLayersWithRoot*2);

       // initialize not deformed cylinder
       for (i = 0 ; i < (2*nrLayersWithRoot); i++)
       {
           cylinderDensity[i]= 1./(2*nrLayersWithRoot);
       }

       // linear and ovoidal deformation
       double deltaDeformation,rootDensitySum;
       rootDensitySum = 0;
       deltaDeformation = deformation - 1;

       for (i = 0 ; i < nrLayersWithRoot; i++)
       {
           cylinderDensity[i] *= deformation;
           deformation -= deltaDeformation/nrLayersWithRoot;
           rootDensitySum += cylinderDensity[i];
       }
       for (i = nrLayersWithRoot; i < (2*nrLayersWithRoot); i++)
       {
           deformation -= deltaDeformation / nrLayersWithRoot;
           cylinderDensity[i] *= deformation;
           rootDensitySum += cylinderDensity[i];
       }
       for (i = nrLayersWithRoot; i < (2*nrLayersWithRoot); i++)
       {
           cylinderDensity[i] /= rootDensitySum;
       }
       for (i = 0; i < totalLayers ; i++)
       {
           densityThinLayers[i] = 0;
       }
       for (i = 0; i < nrLayersWithRoot ; i++)
       {
           densityThinLayers[nrUpperLayersWithoutRoot+i] = cylinderDensity[2*i] + cylinderDensity[2*i+1];
       }
    }


    bool computeRootDensity(Crit3DCrop* myCrop, const std::vector<soil::Crit1DLayer> &soilLayers)
    {
        // check soil
        unsigned int nrLayers = unsigned(soilLayers.size());
        if (nrLayers == 0)
        {
            myCrop->roots.firstRootLayer = NODATA;
            myCrop->roots.lastRootLayer = NODATA;
            return false;
        }

        double soilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

        // Initialize
        for (unsigned int i = 0; i < nrLayers; i++)
        {
            myCrop->roots.rootDensity[i] = 0.0;
        }

        if ((! myCrop->isLiving) || (myCrop->roots.currentRootLength <= 0 ))
            return true;

        if ((myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            || (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION))
        {
            double minimumThickness;
            std::vector<int> atoms;
            atoms.resize(nrLayers);
            int nrAtoms = root::getNrAtoms(soilLayers, minimumThickness, atoms);

            int numberOfRootedLayers, numberOfTopUnrootedLayers;
            numberOfTopUnrootedLayers = int(round(myCrop->roots.rootDepthMin / minimumThickness));
            numberOfRootedLayers = int(round(MINVALUE(myCrop->roots.currentRootLength, soilDepth) / minimumThickness));

            // roots are still too short
            if (numberOfRootedLayers == 0)
                return true;

            // check nr of thin layers
            if ((numberOfTopUnrootedLayers + numberOfRootedLayers) > nrAtoms)
            {
                numberOfRootedLayers = nrAtoms - numberOfTopUnrootedLayers;
            }

            // initialize thin layers density
            std::vector<double> densityThinLayers;
            densityThinLayers.resize(nrAtoms);
            for (int i=0; i < nrAtoms; i++)
            {
                densityThinLayers[i] = 0.;
            }

            if (myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            {
                cardioidDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                     numberOfTopUnrootedLayers, signed(nrAtoms), densityThinLayers);
            }
            else if (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION)
            {
                cylindricalDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                        numberOfTopUnrootedLayers, signed(nrAtoms), densityThinLayers);
            }

            int counter = 0;
            for (unsigned int layer=0; layer < nrLayers; layer++)
            {
                for (int j = 0; j < atoms[layer]; j++)
                {
                    if (counter < nrAtoms)
                        myCrop->roots.rootDensity[layer] += densityThinLayers[counter];
                    counter++;
                }
            }
        }
        else if (myCrop->roots.rootShape == GAMMA_DISTRIBUTION)
        {
            double kappa, theta,a,b;
            double mean, mode;
            mean = myCrop->roots.currentRootLength * 0.5;
            int iterations=0;
            double integralComplementary;
            do{
                mode = 0.6*mean;
                theta = mean - mode;
                kappa = mean / theta;
                iterations++;
                integralComplementary=incompleteGamma(kappa,3*myCrop->roots.currentRootLength/theta) - incompleteGamma(kappa,myCrop->roots.currentRootLength/theta);
                mean *= 0.99;
            } while(integralComplementary>0.01 && iterations<1000);

            for (unsigned int i=1 ; i < nrLayers; i++)
            {
                b = MAXVALUE(soilLayers[i].depth + soilLayers[i].thickness*0.5 - myCrop->roots.rootDepthMin,0); // right extreme
                if (b>0 && b< myCrop->roots.currentRootLength)
                {
                    a = MAXVALUE(soilLayers[i].depth - soilLayers[i].thickness*0.5 - myCrop->roots.rootDepthMin,0); // left extreme
                    myCrop->roots.rootDensity[i] = incompleteGamma(kappa,b/theta) - incompleteGamma(kappa,a/theta); // incompleteGamma is already normalized by gamma(kappa)
                }
                else
                {
                    myCrop->roots.rootDensity[i] = 0;
                }
            }
        }

        double rootDensitySum = 0. ;
        for (unsigned int i=0 ; i < nrLayers; i++)
        {
            myCrop->roots.rootDensity[i] *= soilLayers[i].soilFraction;
            rootDensitySum += myCrop->roots.rootDensity[i];
        }

        if (rootDensitySum > 0.0)
        {
            for (unsigned int i=0 ; i < nrLayers ; i++)
                myCrop->roots.rootDensity[i] /= rootDensitySum;

            myCrop->roots.firstRootLayer = 0;
            unsigned int layer = 0;

            while (layer < nrLayers && myCrop->roots.rootDensity[layer] == 0.0)
            {
                layer++;
                (myCrop->roots.firstRootLayer)++;
            }

            myCrop->roots.lastRootLayer = myCrop->roots.firstRootLayer;
            while (layer < nrLayers && myCrop->roots.rootDensity[layer] != 0.0)
            {
                myCrop->roots.lastRootLayer = signed(layer);
                layer++;
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
            myCrop.roots.firstRootLayer = NODATA;
            myCrop.roots.lastRootLayer = NODATA;
            return false;
        }

        // Initialize
        myCrop.roots.rootDensity.clear();
        myCrop.roots.rootDensity.resize(nrLayers);
        for (unsigned int i = 0; i < nrLayers; i++)
        {
            myCrop.roots.rootDensity[i] = 0.0;
        }

        if (myCrop.roots.currentRootLength <= 0 )
            return true;

        // TODO Gamma distribuion
        if (myCrop.roots.rootShape == GAMMA_DISTRIBUTION)
        {
            myCrop.roots.rootShape = CARDIOID_DISTRIBUTION;
        }

        int nrAtoms = int(currentSoil.totalDepth * 100) + 1;
        double minimumThickness = 0.01;                                    // [m]

        int numberOfRootedLayers, numberOfTopUnrootedLayers;
        numberOfTopUnrootedLayers = int(round(myCrop.roots.rootDepthMin / minimumThickness));
        numberOfRootedLayers = int(round(MINVALUE(myCrop.roots.currentRootLength, currentSoil.totalDepth) / minimumThickness));

        // roots are still too short
        if (numberOfRootedLayers == 0)
            return true;

        // check nr of thin layers
        if ((numberOfTopUnrootedLayers + numberOfRootedLayers) > nrAtoms)
        {
            numberOfRootedLayers = nrAtoms - numberOfTopUnrootedLayers;
        }

        // initialize thin layers density
        std::vector<double> densityThinLayers;
        densityThinLayers.resize(nrAtoms);
        for (int i=0; i < nrAtoms; i++)
        {
            densityThinLayers[i] = 0.;
        }

        if (myCrop.roots.rootShape == CARDIOID_DISTRIBUTION)
        {
            cardioidDistribution(myCrop.roots.shapeDeformation, numberOfRootedLayers,
                                 numberOfTopUnrootedLayers, signed(nrAtoms), densityThinLayers);
        }
        else if (myCrop.roots.rootShape == CYLINDRICAL_DISTRIBUTION)
        {
            cylindricalDistribution(myCrop.roots.shapeDeformation, numberOfRootedLayers,
                                    numberOfTopUnrootedLayers, signed(nrAtoms), densityThinLayers);
        }

        double maxLayerDepth = layerDepth[nrLayers-1] + layerThickness[nrLayers-1] * 0.5;
        int atom = 0;
        double currentDepth = double(atom) * 0.01;                              // [m]
        double rootDensitySum = 0.;
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
            currentDepth = double(atom) * 0.01;                                 // [m]
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
        if (rootDensitySumSubset != rootDensitySum)
        {
            double ratio = rootDensitySum / rootDensitySumSubset;
            for (unsigned int l=0 ; l < nrLayers; l++)
            {
                myCrop.roots.rootDensity[l] *= ratio;
            }
        }

        // find first and last root layers
        myCrop.roots.firstRootLayer = 0;
        unsigned int layer = 0;
        while (layer < nrLayers && myCrop.roots.rootDensity[layer] == 0.0)
        {
            layer++;
            (myCrop.roots.firstRootLayer)++;
        }

        myCrop.roots.lastRootLayer = myCrop.roots.firstRootLayer;
        while (layer < nrLayers && myCrop.roots.rootDensity[layer] != 0.0)
        {
            myCrop.roots.lastRootLayer = signed(layer);
            layer++;
        }

        return true;
    }
}

