/*!
    \copyright 2023
    Fausto Tomei, Gabriele Antolini, Antonio Volta

    This file is part of AGROLIB distribution.
    AGROLIB has been developed under contract issued by A.R.P.A. Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    Contacts:
    ftomei@arpae.it
    gantolini@arpae.it
    avolta@arpae.it
*/

#include <math.h>
#include <float.h>
#include <limits>             // required for LONG_MAX

#include "commonConstants.h"
#include "gammaFunction.h"
#include "basicMath.h"
#include "furtherMathFunctions.h"


    double factorial(int n) {
        static long double const factorials[] = {
               1.000000000000000000000e+0L,          //   0!
               1.000000000000000000000e+0L,          //   1!
               2.000000000000000000000e+0L,          //   2!
               6.000000000000000000000e+0L,          //   3!
               2.400000000000000000000e+1L,          //   4!
               1.200000000000000000000e+2L,          //   5!
               7.200000000000000000000e+2L,          //   6!
               5.040000000000000000000e+3L,          //   7!
               4.032000000000000000000e+4L,          //   8!
               3.628800000000000000000e+5L,          //   9!
               3.628800000000000000000e+6L,          //  10!
               3.991680000000000000000e+7L,          //  11!
               4.790016000000000000000e+8L,          //  12!
               6.227020800000000000000e+9L,          //  13!
               8.717829120000000000000e+10L,         //  14!
               1.307674368000000000000e+12L,         //  15!
               2.092278988800000000000e+13L,         //  16!
               3.556874280960000000000e+14L,         //  17!
               6.402373705728000000000e+15L,         //  18!
               1.216451004088320000000e+17L,         //  19!
               2.432902008176640000000e+18L,         //  20!
               5.109094217170944000000e+19L,         //  21!
               1.124000727777607680000e+21L,         //  22!
               2.585201673888497664000e+22L,         //  23!
               6.204484017332394393600e+23L,         //  24!
               1.551121004333098598400e+25L,         //  25!
               4.032914611266056355840e+26L,         //  26!
               1.088886945041835216077e+28L,         //  27!
               3.048883446117138605015e+29L,         //  28!
               8.841761993739701954544e+30L,         //  29!
               2.652528598121910586363e+32L,         //  30!
               8.222838654177922817726e+33L,         //  31!
               2.631308369336935301672e+35L,         //  32!
               8.683317618811886495518e+36L,         //  33!
               2.952327990396041408476e+38L,         //  34!
               1.033314796638614492967e+40L,         //  35!
               3.719933267899012174680e+41L,         //  36!
               1.376375309122634504632e+43L,         //  37!
               5.230226174666011117600e+44L,         //  38!
               2.039788208119744335864e+46L,         //  39!
               8.159152832478977343456e+47L,         //  40!
               3.345252661316380710817e+49L,         //  41!
               1.405006117752879898543e+51L,         //  42!
               6.041526306337383563736e+52L,         //  43!
               2.658271574788448768044e+54L,         //  44!
               1.196222208654801945620e+56L,         //  45!
               5.502622159812088949850e+57L,         //  46!
               2.586232415111681806430e+59L,         //  47!
               1.241391559253607267086e+61L,         //  48!
               6.082818640342675608723e+62L,         //  49!
               3.041409320171337804361e+64L,         //  50!
               1.551118753287382280224e+66L,         //  51!
               8.065817517094387857166e+67L,         //  52!
               4.274883284060025564298e+69L,         //  53!
               2.308436973392413804721e+71L,         //  54!
               1.269640335365827592597e+73L,         //  55!
               7.109985878048634518540e+74L,         //  56!
               4.052691950487721675568e+76L,         //  57!
               2.350561331282878571829e+78L,         //  58!
               1.386831185456898357379e+80L,         //  59!
               8.320987112741390144276e+81L,         //  60!
               5.075802138772247988009e+83L,         //  61!
               3.146997326038793752565e+85L,         //  62!
               1.982608315404440064116e+87L,         //  63!
               1.268869321858841641034e+89L,         //  64!
               8.247650592082470666723e+90L,         //  65!
               5.443449390774430640037e+92L,         //  66!
               3.647111091818868528825e+94L,         //  67!
               2.480035542436830599601e+96L,         //  68!
               1.711224524281413113725e+98L,         //  69!
               1.197857166996989179607e+100L,        //  70!
               8.504785885678623175212e+101L,        //  71!
               6.123445837688608686152e+103L,        //  72!
               4.470115461512684340891e+105L,        //  73!
               3.307885441519386412260e+107L,        //  74!
               2.480914081139539809195e+109L,        //  75!
               1.885494701666050254988e+111L,        //  76!
               1.451830920282858696341e+113L,        //  77!
               1.132428117820629783146e+115L,        //  78!
               8.946182130782975286851e+116L,        //  79!
               7.156945704626380229481e+118L,        //  80!
               5.797126020747367985880e+120L,        //  81!
               4.753643337012841748421e+122L,        //  82!
               3.945523969720658651190e+124L,        //  83!
               3.314240134565353266999e+126L,        //  84!
               2.817104114380550276949e+128L,        //  85!
               2.422709538367273238177e+130L,        //  86!
               2.107757298379527717214e+132L,        //  87!
               1.854826422573984391148e+134L,        //  88!
               1.650795516090846108122e+136L,        //  89!
               1.485715964481761497310e+138L,        //  90!
               1.352001527678402962552e+140L,        //  91!
               1.243841405464130725548e+142L,        //  92!
               1.156772507081641574759e+144L,        //  93!
               1.087366156656743080274e+146L,        //  94!
               1.032997848823905926260e+148L,        //  95!
               9.916779348709496892096e+149L,        //  96!
               9.619275968248211985333e+151L,        //  97!
               9.426890448883247745626e+153L,        //  98!
               9.332621544394415268170e+155L,        //  99!
               9.332621544394415268170e+157L,        // 100!
               9.425947759838359420852e+159L,        // 101!
               9.614466715035126609269e+161L,        // 102!
               9.902900716486180407547e+163L,        // 103!
               1.029901674514562762385e+166L,        // 104!
               1.081396758240290900504e+168L,        // 105!
               1.146280563734708354534e+170L,        // 106!
               1.226520203196137939352e+172L,        // 107!
               1.324641819451828974500e+174L,        // 108!
               1.443859583202493582205e+176L,        // 109!
               1.588245541522742940425e+178L,        // 110!
               1.762952551090244663872e+180L,        // 111!
               1.974506857221074023537e+182L,        // 112!
               2.231192748659813646597e+184L,        // 113!
               2.543559733472187557120e+186L,        // 114!
               2.925093693493015690688e+188L,        // 115!
               3.393108684451898201198e+190L,        // 116!
               3.969937160808720895402e+192L,        // 117!
               4.684525849754290656574e+194L,        // 118!
               5.574585761207605881323e+196L,        // 119!
               6.689502913449127057588e+198L,        // 120!
               8.094298525273443739682e+200L,        // 121!
               9.875044200833601362412e+202L,        // 122!
               1.214630436702532967577e+205L,        // 123!
               1.506141741511140879795e+207L,        // 124!
               1.882677176888926099744e+209L,        // 125!
               2.372173242880046885677e+211L,        // 126!
               3.012660018457659544810e+213L,        // 127!
               3.856204823625804217357e+215L,        // 128!
               4.974504222477287440390e+217L,        // 129!
               6.466855489220473672507e+219L,        // 130!
               8.471580690878820510985e+221L,        // 131!
               1.118248651196004307450e+224L,        // 132!
               1.487270706090685728908e+226L,        // 133!
               1.992942746161518876737e+228L,        // 134!
               2.690472707318050483595e+230L,        // 135!
               3.659042881952548657690e+232L,        // 136!
               5.012888748274991661035e+234L,        // 137!
               6.917786472619488492228e+236L,        // 138!
               9.615723196941089004197e+238L,        // 139!
               1.346201247571752460588e+241L,        // 140!
               1.898143759076170969429e+243L,        // 141!
               2.695364137888162776589e+245L,        // 142!
               3.854370717180072770522e+247L,        // 143!
               5.550293832739304789551e+249L,        // 144!
               8.047926057471991944849e+251L,        // 145!
               1.174997204390910823948e+254L,        // 146!
               1.727245890454638911203e+256L,        // 147!
               2.556323917872865588581e+258L,        // 148!
               3.808922637630569726986e+260L,        // 149!
               5.713383956445854590479e+262L,        // 150!
               8.627209774233240431623e+264L,        // 151!
               1.311335885683452545607e+267L,        // 152!
               2.006343905095682394778e+269L,        // 153!
               3.089769613847350887959e+271L,        // 154!
               4.789142901463393876336e+273L,        // 155!
               7.471062926282894447084e+275L,        // 156!
               1.172956879426414428192e+278L,        // 157!
               1.853271869493734796544e+280L,        // 158!
               2.946702272495038326504e+282L,        // 159!
               4.714723635992061322407e+284L,        // 160!
               7.590705053947218729075e+286L,        // 161!
               1.229694218739449434110e+289L,        // 162!
               2.004401576545302577600e+291L,        // 163!
               3.287218585534296227263e+293L,        // 164!
               5.423910666131588774984e+295L,        // 165!
               9.003691705778437366474e+297L,        // 166!
               1.503616514864999040201e+300L,        // 167!
               2.526075744973198387538e+302L,        // 168!
               4.269068009004705274939e+304L,        // 169!
               7.257415615307998967397e+306L         // 170!
                                                };

        static const int N = sizeof(factorials) / sizeof(long double);


                   // For a negative argument (n < 0) return 0.0 //

       if ( n < 0 ) return 0.0;


               // For a large postive argument (n >= N) return DBL_MAX //

       if ( n >= N )
       {
           printf("number too big to compute its factorial,; maximal accepted number %d\n", N);
           return DBL_MAX;
       }

                              // Otherwise return n!. //

       return (double) factorials[n];
    }

    double gammaFunction(double value)
    {
        return exp(gammaNaturalLogarithm(value));
    }

    double gammaNaturalLogarithm(double value)
    //Returns the value ln[Γ(xx)] for xx > 0.
    {
    //Internal arithmetic will be done in double precision, a nicety that you can omit if five-figure
    //accuracy is good enough.
        double x,y,tmp,series;
        static double coefficients[6]={76.18009172947146,-86.50532032941677,
        24.01409824083091,-1.231739572450155,
        0.1208650973866179e-2,-0.5395239384953e-5};
        int j;
        y=x=value;
        tmp=x+5.5;
        tmp -= (x+0.5)*log(tmp);
        series=1.000000000190015;
        for (j=0;j<=5;j++) series += coefficients[j]/++y;
        return -tmp+log(2.5066282746310005*series/x);
    }


    void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn)
    //Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser.
    //Also returns ln Γ(a) as gln.
    {
        int n;
        double sum,del,ap;
        *gammaLn=gammaNaturalLogarithm(alpha);
        if (x <= 0.0)
        {
            //if (x < 0.0) printf("x less than 0 in routine gammaIncompleteP");
            *gammaDevelopmentSeries=0.0;
            return;
        }
        else
        {
            ap=alpha;
            del=sum=1.0/alpha;
            for (n=1;n<=ITERATIONSMAX;n++)
            {
                ++ap;
                del *= x/ap;
                sum += del;
                if (fabs(del) < fabs(sum)*EPSTHRESHOLD)
                {
                    *gammaDevelopmentSeries=sum*exp(-x+alpha*log(x)-(*gammaLn));
                    return;
                }
            }
            //printf("a too large, ITERATIONSMAX too small in routine gammaIncompleteP");
            return;
        }
    }

    void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn)
    //Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation
    //as gammcf. Also returns lnΓ(a) as gln.
    {
        int i;
        double an,b,c,d,del,h;
        *gammaLn=gammaNaturalLogarithm(alpha);
        b=x+1.0-alpha; //Set up for evaluating continued fraction by modified Lentz’s method (§5.2)with b0 = 0.
        c=1.0/FPMINIMUM;
        d=1.0/b;
        h=d;
        for (i=1;i<=ITERATIONSMAX;i++)
        { //Iterate to convergence.
            an = -i*(i-alpha);
            b += 2.0;
            d=an*d+b;
            if (fabs(d) < FPMINIMUM) d=FPMINIMUM;
            c=b+an/c;
            if (fabs(c) < FPMINIMUM) c=FPMINIMUM;
            d=1.0/d;
            del=d*c;
            h *= del;
            if (fabs(del-1.0) < EPSTHRESHOLD) break;
        }
        *gammaComplementaryFunction=exp(-x+alpha*log(x)-(*gammaLn))*h; //Put factors in front.
    }

    double incompleteGamma(double alpha, double x, double *lnGammaValue)
    {
        /* this function returns
         * 1) the value of the normalized incomplete gamma function
         * 2) the natural logarithm of the complete gamma function
         * pay attention to the inputs: input variable x is actually beta*x or x/theta
         * written by Antonio Volta avolta@arpae.it
        */
        double gammaIncompleteCF;
        double gammaIncomplete;
        if (x > alpha + 1)
        {
            gammaIncompleteComplementaryFunction(&gammaIncompleteCF,alpha,x,lnGammaValue);
            gammaIncomplete = 1 - gammaIncompleteCF;
        }
        else
            gammaIncompleteP(&gammaIncomplete,alpha,x,lnGammaValue);

        return gammaIncomplete;
    }

    double incompleteGamma(double alpha, double x)
    {
        /* this function returns
         * 1) the value of the normalized incomplete gamma function
         * pay attention to the inputs: input variable x is actually beta*x or x/theta
         * written by Antonio Volta avolta@arpae.it
        */
        double gammaIncompleteCF;
        double gammaIncomplete;
        double lnGammaValue;
        if (x > alpha + 1)
        {
            gammaIncompleteComplementaryFunction(&gammaIncompleteCF,alpha,x,&lnGammaValue);
            gammaIncomplete = 1 - gammaIncompleteCF;
        }
        else
            gammaIncompleteP(&gammaIncomplete,alpha,x,&lnGammaValue);

        return gammaIncomplete;
    }

    bool getGammaParameters(double mean, double variance, double* alpha, double* beta)
    {
        // beta is intended as rate parameter
        if (variance == 0 || mean == 0)
        {
            return false;
        }

        *alpha = variance/mean;
        *beta = mean*mean/variance;
        return true;
    }

    bool generalizedGammaFitting(std::vector<float> &series, int n, double *beta, double *alpha,  double *pZero)
    {
        if (n<=0) return false;

        double sum = 0;
        double sumLog = 0;
        int nAct = 0;
        double average = 0;
        *pZero = 0;

        // compute sums
        for (int i = 0; i<n; i++)
        {
            if (series[i] != NODATA)
            {
                if (series[i] > 0)
                {
                    sum = sum + series[i];
                    sumLog = sumLog + log(series[i]);
                    nAct = nAct + 1;
                }
                else
                {
                    *pZero = *pZero + 1;
                }
            }
        }

        if (nAct > 0)
        {
            average = sum / nAct;
        }

        if (nAct == 1)
        {
            // Bogus data array but do something reasonable
            *pZero = 0;
            *alpha = 1;
            *beta = average;
        }
        else if (*pZero == n)
        {
            // They were all zeroes
            *pZero = 1;
            *alpha = 1;
            *beta = average;
        }
        else
        {
            // Use MLE
            *pZero = *pZero/n;
            double delta = log(average) - sumLog / nAct;
            *alpha = (1 + sqrt(1 + 4 * delta / 3)) / (4 * delta);
            *beta = average / (*alpha);
        }

        if (*alpha <= 0 || *beta <= 0)
        {
            return false;
        }

        return true;
    }


    double standardGaussianInvCDF(double prob)
    {
        double  resul;
        resul = SQRT_2 * statistics::inverseTabulatedERF(2*prob -1);
        return resul;
    }

    /*
    Compute probability of a<=x using incomplete gamma parameters.
    Input:     beta, gamma (gamma parameters)
               pzero (probability of zero)
               x (value)
    Output:    generalizedGammaCDF (probability  a<=x)
    */

    double inverseGammaCumulativeDistributionFunction(double valueProbability, double alpha, double beta, double accuracy)
    {
       double x;
       double y;
       double rightBound = 25.0;
       double leftBound = 0.0;
       int counter = 0;
       do {
           y = incompleteGamma(alpha,rightBound/beta);
           if (valueProbability>y)
           {
               rightBound *= 2;
               counter++;
               if (counter == 7) return rightBound;
           }
       } while ((valueProbability>y));

       x = (rightBound + leftBound)*0.5;
       y = incompleteGamma(alpha,x/beta);
       while ((fabs(valueProbability - y) > accuracy) && (counter < 200))
       {
           if (y > valueProbability)
           {
               rightBound = x;
           }
           else
           {
               leftBound = x;
           }
           x = (rightBound + leftBound)*0.5;
           y = incompleteGamma(alpha,x/beta);
           ++counter;
       }
       x = (rightBound + leftBound)*0.5;
       return x;
    }

    float inverseGeneralizedGammaCDF(float valueProbability, double alpha, double beta, double accuracy,double pZero,double outlierStep)
    {

       if (valueProbability < 0 || valueProbability >= 1)
            return PARAMETER_ERROR;
       if (valueProbability < 0.995)
       {
           float rightBound = 25.0;
           float leftBound = 0.0;
           float x, y;
           int counter = 0;
           do {
               //y = incompleteGamma(alpha,rightBound/beta);
               y = generalizedGammaCDF(rightBound,beta,alpha,pZero);
               if (valueProbability > y)
               {
                   rightBound *= 2;
                   counter++;
                   if (counter == 7) return rightBound;
               }
           } while ((valueProbability > y));

           x = (rightBound + leftBound) * 0.5f;
           y = generalizedGammaCDF(x,beta,alpha,pZero);
           while ((fabs(valueProbability - y) > accuracy) && (counter < 200))
           {
               if (y > valueProbability)
               {
                   rightBound = x;
               }
               else
               {
                   leftBound = x;
               }
               x = (rightBound + leftBound) * 0.5f;
               y = generalizedGammaCDF(x,beta,alpha,pZero);
               ++counter;
           }
           x = (rightBound + leftBound) * 0.5f;
           return x;
       }
       double x,y;
       y = 0.995 - EPSILON;
       x = inverseGeneralizedGammaCDFDoublePrecision(y, alpha, beta, accuracy, pZero,outlierStep);
       while (y < valueProbability)
       {
           y = generalizedGammaCDF(x,beta,alpha,pZero);
           x += outlierStep;
       }
       return float(x - outlierStep);
    }

    double inverseGeneralizedGammaCDFDoublePrecision(double valueProbability, double alpha, double beta, double accuracy,double pZero,double outlierStep)
    {

       if (valueProbability < 0 || valueProbability >= 1)
           return PARAMETER_ERROR;
       if (valueProbability < 0.995)
       {
           double x;
           double y;
           double rightBound = 25.0;
           double leftBound = 0.0;
           int counter = 0;
           do {
               //y = incompleteGamma(alpha,rightBound/beta);
               y = generalizedGammaCDF(rightBound,beta,alpha,pZero);
               if (valueProbability>y)
               {
                   rightBound *= 2;
                   counter++;
                   if (counter == 7) return rightBound;
               }
           } while ((valueProbability>y));

           x = (rightBound + leftBound)*0.5;
           y = generalizedGammaCDF(x,beta,alpha,pZero);
           while ((fabs(valueProbability - y) > accuracy) && (counter < 200))
           {
               if (y > valueProbability)
               {
                   rightBound = x;
               }
               else
               {
                   leftBound = x;
               }
               x = (rightBound + leftBound)*0.5;
               y = generalizedGammaCDF(x,beta,alpha,pZero);
               ++counter;
           }
           x = (rightBound + leftBound)*0.5;
           return x;
       }
       double x,y;
       y = 0.995 - EPSILON;
       x = inverseGeneralizedGammaCDFDoublePrecision(y, alpha, beta, accuracy, pZero,outlierStep);
       while (y < valueProbability)
       {
           y = generalizedGammaCDF(x,beta,alpha,pZero);
           x += outlierStep;
       }
       return x - outlierStep;
    }


    float generalizedGammaCDF(float x, double beta, double alpha,  double pZero)
    {
        if ( isEqual(x, NODATA) || isEqual(beta, NODATA) || isEqual(alpha, NODATA)
            || isEqual(pZero, NODATA) || isEqual(beta, 0) )
        {
            return NODATA;
        }

        double gammaCDF;

        if (x <= 0)
        {
            gammaCDF = pZero;
        }
        else
        {
            gammaCDF = pZero + (1 - pZero) * incompleteGamma(alpha, double(x) / beta);
        }

        return float(gammaCDF);
    }


    double generalizedGammaCDF(double x, double beta, double alpha,  double pZero)
    {

        double gammaCDF = NODATA;

        if (fabs(x - NODATA) < EPSILON || fabs(beta - NODATA)< EPSILON || fabs(alpha - NODATA) < EPSILON || fabs(pZero - NODATA) < EPSILON || beta == 0)
        {
            return gammaCDF;
        }

        if (x <= 0)
        {
            gammaCDF = pZero;
        }
        else
        {
            gammaCDF = pZero + (1 - pZero) * incompleteGamma(alpha, x / beta);
        }
        return gammaCDF;

    }

    float probabilityGamma(float x, double alfa, double gamma, float gammaFunc)
    {
        return float(exp(-alfa * x) *( pow(x,(gamma - 1)) * pow(alfa,gamma) / gammaFunc));
    }

    float probabilityGamma(float x, double alpha, double beta)
    {
        return float(exp(-x/beta) * pow(x,(alpha - 1)) / pow(beta,alpha) / gammaFunction(alpha));
    }

    void probabilityWeightedMoments(std::vector<float> series, int n, std::vector<float> &probWeightedMoments, float a, float b, bool isBeta)
    {

        float f;
        std::vector<float> sum{0,0,0};

        if (a == 0 && b == 0)
        {
            // use unbiased estimator
            for (int i = 1; i <= n; i++)
            {
                sum[0] = sum[0] + series[i-1];
                if (!isBeta)
                {
                    // compute alpha PWMs
                    sum[1] = sum[1] + series[i-1] * (n - i);
                    sum[2] = sum[2] + series[i-1] * (n - i) * (n - i - 1);
                }
                else
                {
                    // compute beta PWMs
                    sum[1] = sum[1] + series[i-1] * (i - 1);
                    sum[2] = sum[2] + series[i-1] * (i - 1) * (i - 2);
                }
            }
        }
        else
        {
            // use plotting-position (biased) estimator
            for (int i = 1; i <= n; i++)
            {
                sum[0] = sum[0] + series[i-1];
                f = (i + a) / (n + b);
                if (!isBeta)
                {
                    // compute alpha PWMs
                    sum[1] = sum[1] + series[i-1] * (1 - f);
                    sum[2] = sum[2] + series[i-1] * (1 - f) * (1 - f);
                }
                else
                {
                    // compute beta PWMs
                    sum[1] = sum[1] + series[i-1] * f;
                    sum[2] = sum[2] + series[i-1] * f * f;
                 }
            }
        }

        probWeightedMoments[0] = sum[0] / n ;
        probWeightedMoments[1] = sum[1] / n / (n - 1) ;
        probWeightedMoments[2] = sum[2] / n / ((n - 1) * (n - 2)) ;

    }

    // Estimates the parameters of a log-logistic distribution function
    void logLogisticFitting(std::vector<float> probWeightedMoments, double *alpha, double *beta, double *gamma)
    {
        double g1, g2;
        *gamma = (2 * probWeightedMoments[1] - probWeightedMoments[0]) / (6 * probWeightedMoments[1] - probWeightedMoments[0] - 6 * probWeightedMoments[2]);
        //g1 = exp(Ln_Gamma_Function(1 + 1 / (*gamma)));
        //g2 = exp(Ln_Gamma_Function(1 - 1 / (*gamma)));
        g1 = gammaFunction(1 + 1 / (*gamma));
        g2 = gammaFunction(1 - 1 / (*gamma));
        *alpha = (probWeightedMoments[0] - 2 * probWeightedMoments[1]) * (*gamma) / (g1 * g2);
        *beta = probWeightedMoments[0] - (*alpha) * g1 * g2;
    }

    // Gives the cumulative distribution function of input "myValue",
    // following a LogLogistic distribution
    float logLogisticCDF(float myValue, double alpha, double beta, double gamma)
    {
        double logLogisticCDF = 1. / (1. + (pow((alpha / (double(myValue) - beta)), gamma)));

        return float(logLogisticCDF);
    }

    double weibullCDF(double x, double lambda, double kappa)
    {
        double value;
        value = 1 - exp(-pow((x/lambda),kappa));
        return value;
    }

    double inverseWeibullCDF(double x, double lambda, double kappa)
    {
        double value;
        if (x >= 1  || x < 0 || kappa <= 0 || lambda <= 0) return PARAMETER_ERROR;
        value = lambda*pow(-log(1-x),1./kappa);
        return value;
    }

    double weibullPDF(double x, double lambda, double kappa)
    {
        double value;
        value = (kappa/pow(lambda,kappa))*pow(x,kappa-1)*exp(-pow((x/lambda),kappa));
        return value;
    }

    double meanValueWeibull(double lambda, double kappa)
    {
        double value;
        value = (lambda/kappa)*gammaFunction(1./kappa);
        return value;
    }

    double varianceValueWeibull(double lambda, double kappa)
    {
        double value;
        value = 2.*lambda*lambda/kappa*gammaFunction(2./kappa) - pow(meanValueWeibull(lambda,kappa),2);
        return value;
    }

    double functionValueVarianceWeibullDependingOnKappa(double mean, double variance, double kappa)
    {
        double func;
        func = 2*mean*mean*(kappa)/pow(gammaFunction(1./(kappa)),2)*gammaFunction(2./(kappa))- mean*mean - variance;
        return func;
    }

    void parametersWeibullFromObservations(double mean, double variance, double* lambda, double* kappa, double leftBound, double rightBound)
    {
        double rightK = rightBound;
        double leftK = leftBound;
        double funcRight,funcLeft;
        funcRight = functionValueVarianceWeibullDependingOnKappa(mean,variance,rightK);
        funcLeft = functionValueVarianceWeibullDependingOnKappa(mean,variance,leftK);
        int counter = 0;
        while (funcRight*funcLeft > 0 && counter < 1000)
        {
            rightK += 0.01;
            leftK -= 0.01;
            if (leftK < 0.05) leftK = 0.05;
            funcRight = functionValueVarianceWeibullDependingOnKappa(mean,variance,rightK);
            funcLeft = functionValueVarianceWeibullDependingOnKappa(mean,variance,leftK);
            counter++;
        }
        counter = 0;
        double precision = 0.001;
        double k=(rightK+leftK)*0.5;
        double func;
        double deltaFunc = fabs(funcLeft-funcRight);
        double deltaK = rightK - leftK;
        while (deltaK> 0.00001 && deltaFunc > precision && counter < 10000)
        {
            func = functionValueVarianceWeibullDependingOnKappa(mean,variance,k);
            if (funcRight*func >0)
            {
                rightK = k;
                funcRight = func;
            }
            else
            {
                leftK = k;
                funcLeft = func;
            }
            deltaFunc = fabs(funcLeft-funcRight);
            k = (rightK + leftK)*0.5;
            deltaK = rightK - leftK;
            counter++;
        }
        *kappa = k;
        *lambda = mean*k/gammaFunction(1./k);
    }
