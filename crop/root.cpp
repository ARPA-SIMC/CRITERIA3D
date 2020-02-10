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
#include <iostream>

#include "commonConstants.h"
#include "gammaFunction.h"
#include "root.h"
#include "crop.h"
#include "soil.h"


Crit3DRoot::Crit3DRoot()
{
    this->rootShape = CYLINDRICAL_DISTRIBUTION;
    this->growth = LOGISTIC;
    this->shapeDeformation = NODATA;
    this->degreeDaysRootGrowth = NODATA;
    this->rootDepthMin = NODATA;
    this->rootDepthMax = NODATA;
    this->firstRootLayer = NODATA;
    this->lastRootLayer = NODATA;
    this->rootLength = NODATA;
    this->rootDepth = NODATA;
    this->rootDensity = nullptr;
    this->transpiration = nullptr;
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
                    return GAMMA_DISTRIBUTION;
             }
        }

    double computeRootDepth(Crit3DCrop* myCrop, double soilDepth, double currentDD, double waterTableDepth)
    {
        if (!(myCrop->isLiving))
        {
            myCrop->roots.rootLength = 0.0;
            myCrop->roots.rootDepth = NODATA;
        }
        else
        {
            myCrop->roots.rootLength = computeRootLength(myCrop, soilDepth, currentDD, waterTableDepth);
            myCrop->roots.rootDepth = myCrop->roots.rootDepthMin + myCrop->roots.rootLength;
        }

        return myCrop->roots.rootDepth;
    }


    // TODO this function computes the root length based on thermal units, it could be changed for perennial crops
    double computeRootLength(Crit3DCrop* myCrop, double soilDepth, double currentDD, double waterTableDepth)
    {
        double myRootLength = NODATA;

        if (myCrop->roots.rootDepthMax > soilDepth)
        {
            myCrop->roots.rootDepthMax = soilDepth; // attenzione è diverso da criteria
            std::cout << "Warning: input root profile deeper than soil profile\n";
        }

        if (myCrop->isPluriannual())
        {
            myRootLength = myCrop->roots.rootDepthMax - myCrop->roots.rootDepthMin;
        }
        else
        {
            if (currentDD <= 0)
                myRootLength = 0.0;
            else if (currentDD > myCrop->roots.degreeDaysRootGrowth)
                myRootLength = myCrop->roots.rootDepthMax - myCrop->roots.rootDepthMin;
            else
            {
                // in order to avoid numerical divergences when calculating density through cardioid and gamma function
                currentDD = MAXVALUE(currentDD, 1.0);
                myRootLength = getRootLengthDD(&(myCrop->roots), currentDD, myCrop->degreeDaysEmergence);
            }
        }

        // WATERTABLE
        // le radici nel terreno saturo vanno in asfissia
        // per cui vanno mantenute a distanza nella fase di crescita
        // le radici possono crescere se:
        // la falda è più bassa o si abbassa (max 2 cm al giorno)
        // restano invariate se:
        // 1) non sono più in fase di crescita
        // 2) se sono già dentro la falda
        const double MAX_DAILY_GROWTH = 0.02;             // [m]
        const double MIN_WATERTABLE_DISTANCE = 0.2;       // [m]

        if (int(waterTableDepth) != int(NODATA)
                && waterTableDepth > 0
                && int(myCrop->roots.rootLength) != int(NODATA)
                && !myCrop->isWaterSurplusResistant()
                && myRootLength > myCrop->roots.rootLength)
        {
            // check on growth
            if (currentDD > myCrop->roots.degreeDaysRootGrowth)
                myRootLength = myCrop->roots.rootLength;
            else
                myRootLength = MINVALUE(myRootLength, myCrop->roots.rootLength + MAX_DAILY_GROWTH);

            // check on watertable
            double maxLenght = waterTableDepth - myCrop->roots.rootDepthMin - MIN_WATERTABLE_DISTANCE;
            if (myRootLength > maxLenght)
            {
                myRootLength = MAXVALUE(myCrop->roots.rootLength, maxLenght);
            }
        }

        return myRootLength;
    }


    //[m]
    double getRootLengthDD(Crit3DRoot* myRoot, double currentDD, double emergenceDD)
    {
        // this function computes the roots rate of development
        double myRootLength = NODATA;
        double maxRootLength = myRoot->rootDepthMax - myRoot->rootDepthMin;

        if (currentDD <= 0) return 0.;
        if (currentDD > myRoot->degreeDaysRootGrowth) return maxRootLength;

        double halfDevelopmentPoint = myRoot->degreeDaysRootGrowth * 0.5 ;

        if (myRoot->growth == LINEAR)
        {
            myRootLength = maxRootLength * (currentDD / myRoot->degreeDaysRootGrowth);
        }
        else if (myRoot->growth == LOGISTIC)
        {
            double logMax, logMin,deformationFactor;
            double iniLog = log(9.);
            double filLog = log(1 / 0.99 - 1);
            double k,b;
            k = -(iniLog - filLog) / (emergenceDD - myRoot->degreeDaysRootGrowth);
            b = -(filLog + k * myRoot->degreeDaysRootGrowth);

            logMax = (myRoot->rootDepthMax) / (1 + exp(-b - k * myRoot->degreeDaysRootGrowth));
            logMin = myRoot->rootDepthMax / (1 + exp(-b));
            deformationFactor = (logMax - logMin) / maxRootLength ;
            myRootLength = 1.0 / deformationFactor * (myRoot->rootDepthMax / (1.0 + exp(-b - k * currentDD)) - logMin);
        }
        else if (myRoot->growth == EXPONENTIAL)
        {
            // not used in Criteria Bdp
            myRootLength = maxRootLength * (1.- exp(-2.*(currentDD/halfDevelopmentPoint)));
        }

        return myRootLength;
    }


    int highestCommonFactor(int* vector, int vectorDim)
    {
        // highest common factor (hcf) amongst n integer numbers
        int num1, num2, i, hcf;
        hcf = num1 = vector[0];
        for (int j=0; j<vectorDim-1; j++)
        {

            num1 = hcf;
            num2 = vector[j+1];

            for(i=1; i<=num1 || i<=num2; ++i)
            {
                if(num1%i==0 && num2%i==0)   /* Checking whether i is a factor of both number */
                    hcf=i;
            }
        }
        return hcf;
    }

    int checkTheOrderOfMagnitude(double number,int* order)
    {

        if (number<1)
        {
            number *= 10;
            (*order)--;
            checkTheOrderOfMagnitude(number,order);
        }
        else if (number >= 10)
        {
            number /=10;
            (*order)++;
            checkTheOrderOfMagnitude(number,order);
        }
        return 0;
    }

    int orderOfMagnitude(double number)
    {
        int order = 0;
        number = fabs(number);
        checkTheOrderOfMagnitude(number,&order);
        return order;
    }

    int nrAtoms(soil::Crit3DLayer* layers, int nrLayers, double rootDepthMin, double* minThickness, int* atoms)
    {
        int multiplicationFactor = 1;

        if (rootDepthMin > 0)
            *minThickness = rootDepthMin;
        else
            *minThickness = layers[1].thickness;

        for(int i=1; i<nrLayers; i++)
            *minThickness = MINVALUE(*minThickness, layers[i].thickness);

        double tmp = *minThickness * 1.001;
        if (tmp < 1)
            multiplicationFactor = int(pow(10.0,-orderOfMagnitude(tmp)));

        if (*minThickness < 1)
        {
            *minThickness = 1./multiplicationFactor;
        }

        int value;
        int counter = 0;
        for(int i=0; i<nrLayers; i++)
        {
           value = int(round(multiplicationFactor * layers[i].thickness));
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
    void cardioidDistribution(double shapeFactor, int nrLayersWithRoot,
                              int nrUpperLayersWithoutRoot , int totalLayers, double* densityThinLayers)
    {
        double *lunette =  new double[unsigned(2*nrLayersWithRoot)];
        double *lunetteDensity = new double[unsigned(2*nrLayersWithRoot)];
        for (int i = 0 ; i<nrLayersWithRoot ; i++)
        {
            double sinAlfa, cosAlfa, alfa;
            sinAlfa = 1. - double(1.+i)/(double(nrLayersWithRoot));
            cosAlfa = MAXVALUE(sqrt(1. - pow(sinAlfa,2)), 0.0001);
            alfa = atan(sinAlfa/cosAlfa);
            lunette[i]= ((PI/2) - alfa - sinAlfa*cosAlfa) / PI;
        }

        lunetteDensity[2*nrLayersWithRoot - 1]= lunetteDensity[0] = lunette[0];
        for (int i = 1 ; i<nrLayersWithRoot ; i++)
        {
            lunetteDensity[2*nrLayersWithRoot - i - 1]=lunetteDensity[i]=lunette[i]-lunette[i-1];
        }

        // cardioid deformation
        double LiMin,Limax,k,rootDensitySum ;
        LiMin = -log(0.2) / nrLayersWithRoot;
        Limax = -log(0.05) / nrLayersWithRoot;
        // TODO verify
        k = LiMin + (Limax - LiMin) * (shapeFactor-1);
        rootDensitySum = 0 ;
        for (int i = 0 ; i<(2*nrLayersWithRoot) ; i++)
        {
            lunetteDensity[i] *= exp(-k*(i+0.5));
            rootDensitySum += lunetteDensity[i];
        }
        for (int i = 0 ; i<(2*nrLayersWithRoot) ; i++)
        {
            lunetteDensity[i] /= rootDensitySum ;
        }
        for  (int i = 0 ; i < totalLayers ; i++)
        {
            densityThinLayers[i]=0;
        }
        for (int i = 0 ; i<nrLayersWithRoot ; i++)
        {
            densityThinLayers[nrUpperLayersWithoutRoot+i] = lunetteDensity[2*i] + lunetteDensity[2*i+1];
        }

        free(lunette);
        free(lunetteDensity);
    }


    void cylindricalDistribution(double deformation, int nrLayersWithRoot,int nrUpperLayersWithoutRoot , int totalLayers,double* densityThinLayers)
    {
       int i;

       double *cylinderDensity =  new double[unsigned(2*nrLayersWithRoot)];
       for (i = 0 ; i<2*nrLayersWithRoot; i++)
       {
           cylinderDensity[i]= 1./(2*nrLayersWithRoot);
       } // not deformed cylinder

       // linear and ovoidal deformation
       double deltaDeformation,rootDensitySum;
       rootDensitySum =0;
       deltaDeformation = deformation - 1;

       for (i = 0 ; i<nrLayersWithRoot ; i++)
       {
           cylinderDensity[i] *= deformation;
           deformation -= deltaDeformation/nrLayersWithRoot;
           rootDensitySum += cylinderDensity[i];
       }
       for (i = nrLayersWithRoot ; i<2*nrLayersWithRoot ; i++)
       {
           deformation -= deltaDeformation / nrLayersWithRoot;
           cylinderDensity[i] *= deformation;
           rootDensitySum += cylinderDensity[i];
       }
       for (i = nrLayersWithRoot ; i<2*nrLayersWithRoot ; i++)
       {
           cylinderDensity[i] /= rootDensitySum;
       }
       for (i = 0 ; i<totalLayers ; i++)
       {
           densityThinLayers[i] = 0;
       }
       for (i = 0 ; i<nrLayersWithRoot ; i++)
       {
           densityThinLayers[nrUpperLayersWithoutRoot+i] = cylinderDensity[2*i] + cylinderDensity[2*i+1] ;
       }
       free(cylinderDensity);
    }


    bool computeRootDensity(Crit3DCrop* myCrop, soil::Crit3DLayer* layers, int nrLayers, double soilDepth)
    {
        int i, layer;

        // Initialize
        for (i = 0; i < nrLayers; i++)
        {
            myCrop->roots.rootDensity[i] = 0.0;
        }

        if ((! myCrop->isLiving) || (myCrop->roots.rootLength <= 0 )) return true;

        if ((myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            || (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION))
        {
            double minimumThickness;
            int *atoms = new int[unsigned(nrLayers)];
            int numberOfRootedLayers, numberOfTopUnrootedLayers, totalLayers;
            totalLayers = root::nrAtoms(layers, nrLayers, myCrop->roots.rootDepthMin, &minimumThickness, atoms);
            numberOfTopUnrootedLayers = int(round(myCrop->roots.rootDepthMin / minimumThickness));
            numberOfRootedLayers = int(ceil(MINVALUE(myCrop->roots.rootLength, soilDepth) / minimumThickness));
            double *densityThinLayers =  new double[unsigned(totalLayers+1)];
            densityThinLayers[totalLayers] = 0.;
            for (i=0; i < totalLayers; i++)
                densityThinLayers[i] = 0.;

            if (myCrop->roots.rootShape == CARDIOID_DISTRIBUTION)
            {
                cardioidDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                     numberOfTopUnrootedLayers, totalLayers, densityThinLayers);
            }
            else if (myCrop->roots.rootShape == CYLINDRICAL_DISTRIBUTION)
            {
                cylindricalDistribution(myCrop->roots.shapeDeformation, numberOfRootedLayers,
                                        numberOfTopUnrootedLayers, totalLayers, densityThinLayers);
            }

            int j, counter=0;
            for (layer=0; layer<nrLayers; layer++)
            {
                for (j = counter; j < (counter + atoms[layer]); j++)
                {
                    if (j < totalLayers)
                        myCrop->roots.rootDensity[layer] += densityThinLayers[j];
                }
                counter = j;
            }
            free(atoms);
            free(densityThinLayers);
        }
        else if (myCrop->roots.rootShape == GAMMA_DISTRIBUTION)
        {
            double normalizationFactor ;
            double kappa, theta,a,b;
            double mean, mode;
            mean = myCrop->roots.rootLength * 0.5;
            mode = myCrop->roots.rootLength * 0.2;
            theta = mean - mode;
            kappa = mean / theta;
            // complete gamma function
            normalizationFactor = Gamma_Function(kappa);

            for (i=1 ; i<nrLayers ; i++)
            {
                b = MAXVALUE(layers[i].depth + layers[i].thickness*0.5 - myCrop->roots.rootDepthMin,0); // right extreme
                if (b>0)
                {
                    a = MAXVALUE(layers[i].depth - layers[i].thickness*0.5 - myCrop->roots.rootDepthMin,0); //left extreme
                    myCrop->roots.rootDensity[i] = Incomplete_Gamma_Function(b/theta,kappa) - Incomplete_Gamma_Function(a/theta,kappa);
                    myCrop->roots.rootDensity[i] /= normalizationFactor;
                }
            }
        }

        double rootDensitySum = 0. ;
        for (i=0 ; i<nrLayers ; i++)
        {
            myCrop->roots.rootDensity[i] *= layers[i].soilFraction;
            rootDensitySum += myCrop->roots.rootDensity[i];
        }

        if (rootDensitySum > 0.0)
        {
            for (i=0 ; i<nrLayers ; i++)
                myCrop->roots.rootDensity[i] /= rootDensitySum;

            myCrop->roots.firstRootLayer = 0;
            layer = 0;

            while (layer < nrLayers && myCrop->roots.rootDensity[layer] == 0.0)
            {
                layer++;
                (myCrop->roots.firstRootLayer)++;
            }

            myCrop->roots.lastRootLayer = myCrop->roots.firstRootLayer;
            while (layer < nrLayers && myCrop->roots.rootDensity[layer] != 0.0)
            {
                (myCrop->roots.lastRootLayer) = layer;
                layer++;
            }
        }

        return true;
    }
}

