#include <math.h>

#include "commonConstants.h"
#include "wgClimate.h"
#include "weatherGenerator.h"
#include "timeUtility.h"
#include "crit3dDate.h"
#include <iostream>
#include <QFile>
#include <QTextStream>

using namespace std;


/*!
  * \brief Compute climate (monthly values)
  * \returns true if the input data are valid
  * \param  nrDays          [-] number of data (366 x n where n is the number of years)
  * \param  inputFirstDate  [Crit3DDate]
  * \param  *inputTMin      [°C] array(1..nrDays) of minimum temperature
  * \param  *inputTMax      [°C] array(1..nrDays) of maximum temperature
  * \param  *inputPrec      [mm] array(1..nrDays) of precipitation
*/
bool computeWGClimate(int nrDays, Crit3DDate inputFirstDate, const std::vector<float>& inputTMin,
                      const std::vector<float>& inputTMax, const std::vector<float>& inputPrec,
                      float precThreshold, float minPrecData,
                      TweatherGenClimate* wGen, bool writeOutput, QString outputFileName)
{
    long nValidData = 0;
    float dataPresence = 0;
    double sumTMin[12] = {0};
    double sumTMax[12] = {0};
    double sumPrec[12] = {0};
    double sumTMin2[12] = {0};
    double sumTMax2[12] = {0};
    long nWetDays[12] = {0};
    long nWetWetDays[12] = {0};
    long nDryDays[12] = {0};
    long nrData[12] = {0};
    double sumTmaxWet[12] = {0};
    double sumTmaxDry[12] = {0};
    double sumTminWet[12] = {0};
    double sumTminDry[12] = {0};
    int daysInMonth;
    bool isPreviousDayWet = false;

    // read data
    int m;
    Crit3DDate myDate = inputFirstDate;
    for (int n = 0; n < nrDays; n++)
    {
        m = myDate.month - 1;

        // the day is valid if all values are different from nodata
        if (int(inputTMin[n]) != int(NODATA)
            && int(inputTMax[n]) != int(NODATA)
            && int(inputPrec[n]) != int(NODATA))
        {
            nValidData++;
            nrData[m]++;
            sumTMin[m] += double(inputTMin[n]);
            sumTMin2[m] += double(inputTMin[n] * inputTMin[n]);
            sumTMax[m] += double(inputTMax[n]);
            sumTMax2[m] += double(inputTMax[n] * inputTMax[n]);
            sumPrec[m] += double(inputPrec[n]);

            if (inputPrec[n] > precThreshold)
            {
                if (isPreviousDayWet) nWetWetDays[m]++;
                nWetDays[m]++;
                sumTmaxWet[m] += double(inputTMax[n]);
                sumTminWet[m] += double(inputTMin[n]);
                isPreviousDayWet = true;
            }
            else
            {
                nDryDays[m]++;
                sumTmaxDry[m] += double(inputTMax[n]);
                sumTminDry[m] += double(inputTMin[n]);
                isPreviousDayWet = false;
            }
        }

        ++myDate;
    }

    dataPresence = float(nValidData) / float(nrDays);
    if (dataPresence < minPrecData)
        return false;

    // compute Climate
    for (m=0; m<12; m++)
    {
        if (nrData[m] > 0)
        {
            wGen->monthly.monthlyTmax[m] = sumTMax[m] / nrData[m]; //computes mean monthly values of maximum temperature
            wGen->monthly.monthlyTmin[m] = sumTMin[m] / nrData[m]; //computes mean monthly values of minimum temperature
            wGen->monthly.monthlyTmaxDry[m] = sumTmaxDry[m] / nDryDays[m];
            wGen->monthly.monthlyTmaxWet[m] = sumTmaxWet[m] / nWetDays[m];
            wGen->monthly.monthlyTminDry[m] = sumTminDry[m] / nDryDays[m];
            wGen->monthly.monthlyTminWet[m] = sumTminWet[m] / nWetDays[m];

            daysInMonth = getDaysInMonth(m+1,2001); // year = 2001 is to avoid leap year

            wGen->monthly.sumPrec[m] = sumPrec[m] / nrData[m] * daysInMonth;

            wGen->monthly.fractionWetDays[m] = float(nWetDays[m]) / float(nrData[m]);
            wGen->monthly.probabilityWetWet[m] = float(nWetWetDays[m]) / float(nWetDays[m]);

            if ( (nDryDays[m] > 0) && (nWetDays[m] > 0) )
                wGen->monthly.dw_Tmax[m] = (sumTmaxDry[m] / nDryDays[m]) - (sumTmaxWet[m] / nWetDays[m]);
            else
                wGen->monthly.dw_Tmax[m] = 0;

            wGen->monthly.stDevTmax[m] = sqrt(MAXVALUE(nrData[m]*sumTMax2[m]-(sumTMax[m]*sumTMax[m]), 0) / (nrData[m]*(nrData[m]-1)));
            wGen->monthly.stDevTmin[m] = sqrt(MAXVALUE(nrData[m]*sumTMin2[m]-(sumTMin[m]*sumTMin[m]), 0) / (nrData[m]*(nrData[m]-1)));
        }
        else
        {
            wGen->monthly.monthlyTmax[m] = NODATA;
            wGen->monthly.monthlyTmin[m] = NODATA;
            wGen->monthly.sumPrec[m] = NODATA;
            wGen->monthly.fractionWetDays[m] = NODATA;
            wGen->monthly.stDevTmax[m] = NODATA;
            wGen->monthly.stDevTmin[m] = NODATA;
            wGen->monthly.dw_Tmax[m] = NODATA;
        }
    }


    if (writeOutput)
    {
        cout << "...Write WG climate file -->" << outputFileName.toStdString() << "\n";

        QFile file(outputFileName);
        file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);

        QTextStream stream( &file );
        stream << "----------------- CLIMATE ----------------\n";
        for (m=0; m<12; m++)
        {
            stream << "month = " << m +1 << "\n";
            stream << "wGen->monthly.monthlyTmin = " << wGen->monthly.monthlyTmin[m] << "\n";
            stream << "wGen->monthly.monthlyTmax = " << wGen->monthly.monthlyTmax[m] << "\n";
            stream << "wGen->monthly.sumPrec = " << wGen->monthly.sumPrec[m] << "\n";
            stream << "wGen->monthly.stDevTmin = " << wGen->monthly.stDevTmin[m] << "\n";
            stream << "wGen->monthly.stDevTmax = " << wGen->monthly.stDevTmax[m] << "\n";
            stream << "wGen->monthly.fractionWetDays = " << wGen->monthly.fractionWetDays[m] << "\n";
            stream << "wGen->monthly.probabilityWetWet = " << wGen->monthly.probabilityWetWet[m] << "\n";
            stream << "wGen->monthly.dw_Tmax = " << wGen->monthly.dw_Tmax[m] << "\n";
            stream << "wGen->monthly.monthlyTminDry = " << wGen->monthly.monthlyTminDry[m] << "\n";
            stream << "wGen->monthly.monthlyTmaxDry = " << wGen->monthly.monthlyTmaxDry[m] << "\n";
            stream << "wGen->monthly.monthlyTminWet = " << wGen->monthly.monthlyTminWet[m] << "\n";
            stream << "wGen->monthly.monthlyTmaxWet = " << wGen->monthly.monthlyTmaxWet[m] << "\n";

            stream << "-------------------------------------------" << "\n";
        }
    }

    return true;
}


bool computeWG2DClimate(int nrDays, Crit3DDate inputFirstDate, float *inputTMin, float *inputTMax,
                        float *inputPrec, float precThreshold, float minPrecData,
                        TweatherGenClimate* wGen, bool writeOutput,bool outputForStats, QString outputFileName,
                        float* monthlyPrecipitation, float** consecutiveDry, float** consecutiveWet,
                        int nrConsecutiveDryDaysBins)
{
    long nValidData = 0;
    float dataPresence = 0;
    double sumTMin[12] = {0};
    double sumTMax[12] = {0};
    double sumPrec[12] = {0};
    double sumTMin2[12] = {0};
    double sumTMax2[12] = {0};
    long nWetDays[12] = {0};
    long nWetWetDays[12] = {0};
    long nDryDays[12] = {0};
    long nrData[12] = {0};
    double sumTmaxWet[12] = {0};
    double sumTmaxDry[12] = {0};
    double sumTminWet[12] = {0};
    double sumTminDry[12] = {0};
    double sumTmaxWet2[12] = {0};
    double sumTmaxDry2[12] = {0};
    double sumTminWet2[12] = {0};
    double sumTminDry2[12] = {0};
    int daysInMonth;
    bool isPreviousDayWet = false;

    // read data
    int m;
    Crit3DDate myDate = inputFirstDate;
    for (int n = 0; n < nrDays; n++)
    {
        m = myDate.month - 1;

        // the day is valid if all values are different from nodata
        if (int(inputTMin[n]) != int(NODATA)
            && int(inputTMax[n]) != int(NODATA)
            && int(inputPrec[n]) != int(NODATA))
        {
            nValidData++;
            nrData[m]++;
            sumTMin[m] += double(inputTMin[n]);
            sumTMin2[m] += double(inputTMin[n] * inputTMin[n]);
            sumTMax[m] += double(inputTMax[n]);
            sumTMax2[m] += double(inputTMax[n] * inputTMax[n]);
            sumPrec[m] += double(inputPrec[n]);

            if (inputPrec[n] > precThreshold)
            {
                if (isPreviousDayWet) nWetWetDays[m]++;
                nWetDays[m]++;
                sumTmaxWet[m] += double(inputTMax[n]);
                sumTminWet[m] += double(inputTMin[n]);
                sumTmaxWet2[m] += double(inputTMax[n] * inputTMax[n]);
                sumTminWet2[m] += double(inputTMin[n] * inputTMin[n]);
                isPreviousDayWet = true;
            }
            else
            {
                nDryDays[m]++;
                sumTmaxDry[m] += double(inputTMax[n]);
                sumTminDry[m] += double(inputTMin[n]);
                sumTmaxDry2[m] += double(inputTMax[n] * inputTMax[n]);
                sumTminDry2[m] += double(inputTMin[n] * inputTMin[n]);
                isPreviousDayWet = false;
            }
        }

        ++myDate;
    }

    dataPresence = float(nValidData) / float(nrDays);
    if (dataPresence < minPrecData)
        return false;

    // compute Climate
    for (m=0; m<12; m++)
    {
        if (nrData[m] > 0)
        {
            wGen->monthly.monthlyTmax[m] = sumTMax[m] / nrData[m]; //computes mean monthly values of maximum temperature
            wGen->monthly.monthlyTmin[m] = sumTMin[m] / nrData[m]; //computes mean monthly values of minimum temperature
            wGen->monthly.monthlyTmaxDry[m] = sumTmaxDry[m] / nDryDays[m];
            wGen->monthly.monthlyTmaxWet[m] = sumTmaxWet[m] / nWetDays[m];
            wGen->monthly.monthlyTminDry[m] = sumTminDry[m] / nDryDays[m];
            wGen->monthly.monthlyTminWet[m] = sumTminWet[m] / nWetDays[m];

            daysInMonth = getDaysInMonth(m+1,2001); // year = 2001 is to avoid leap year

            wGen->monthly.sumPrec[m] = sumPrec[m] / nrData[m] * daysInMonth;

            wGen->monthly.fractionWetDays[m] = float(nWetDays[m]) / float(nrData[m]);
            wGen->monthly.probabilityWetWet[m] = float(nWetWetDays[m]) / float(nWetDays[m]);

            if ( (nDryDays[m] > 0) && (nWetDays[m] > 0) )
                wGen->monthly.dw_Tmax[m] = (sumTmaxDry[m] / nDryDays[m]) - (sumTmaxWet[m] / nWetDays[m]);
            else
                wGen->monthly.dw_Tmax[m] = 0;

            wGen->monthly.stDevTmax[m] = sqrt(MAXVALUE(nrData[m]*sumTMax2[m]-(sumTMax[m]*sumTMax[m]), 0) / (nrData[m]*(nrData[m]-1)));
            wGen->monthly.stDevTmin[m] = sqrt(MAXVALUE(nrData[m]*sumTMin2[m]-(sumTMin[m]*sumTMin[m]), 0) / (nrData[m]*(nrData[m]-1)));
            wGen->monthly.stDevTmaxDry[m] = sqrt(MAXVALUE(nDryDays[m]*sumTmaxDry2[m]-(sumTmaxDry[m]*sumTmaxDry[m]), 0) / (nDryDays[m]*(nDryDays[m]-1)));
            wGen->monthly.stDevTmaxWet[m] = sqrt(MAXVALUE(nWetDays[m]*sumTmaxWet2[m]-(sumTmaxWet[m]*sumTmaxWet[m]), 0) / (nWetDays[m]*(nWetDays[m]-1)));
            wGen->monthly.stDevTminDry[m] = sqrt(MAXVALUE(nDryDays[m]*sumTminDry2[m]-(sumTminDry[m]*sumTminDry[m]), 0) / (nDryDays[m]*(nDryDays[m]-1)));
            wGen->monthly.stDevTminWet[m] = sqrt(MAXVALUE(nWetDays[m]*sumTminWet2[m]-(sumTminWet[m]*sumTminWet[m]), 0) / (nWetDays[m]*(nWetDays[m]-1)));

        }
        else
        {
            wGen->monthly.monthlyTmax[m] = NODATA;
            wGen->monthly.monthlyTmin[m] = NODATA;
            wGen->monthly.sumPrec[m] = NODATA;
            wGen->monthly.fractionWetDays[m] = NODATA;
            wGen->monthly.stDevTmax[m] = NODATA;
            wGen->monthly.stDevTmin[m] = NODATA;
            wGen->monthly.dw_Tmax[m] = NODATA;
            wGen->monthly.stDevTmaxDry[m] = NODATA;
            wGen->monthly.stDevTminDry[m] = NODATA;
            wGen->monthly.stDevTmaxWet[m] = NODATA;
            wGen->monthly.stDevTminWet[m] = NODATA;
        }
        monthlyPrecipitation[m] = wGen->monthly.sumPrec[m];
    }


    if (writeOutput)
    {
        if (!outputForStats)
        {
            cout << "...Write WG climate file -->" << outputFileName.toStdString() << "\n";

            QFile file(outputFileName);
            file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);

            QTextStream stream( &file );
            stream << "----------------- CLIMATE ----------------\n";
            for (m=0; m<12; m++)
            {
                stream << "month = " << m +1 << "\n";
                stream << "wGen->monthly.monthlyTmin = " << wGen->monthly.monthlyTmin[m] << "\n";
                stream << "wGen->monthly.monthlyTmax = " << wGen->monthly.monthlyTmax[m] << "\n";
                stream << "wGen->monthly.sumPrec = " << wGen->monthly.sumPrec[m] << "\n";
                stream << "wGen->monthly.stDevTmin = " << wGen->monthly.stDevTmin[m] << "\n";
                stream << "wGen->monthly.stDevTmax = " << wGen->monthly.stDevTmax[m] << "\n";
                stream << "wGen->monthly.fractionWetDays = " << wGen->monthly.fractionWetDays[m] << "\n";
                stream << "wGen->monthly.probabilityWetWet = " << wGen->monthly.probabilityWetWet[m] << "\n";
                stream << "wGen->monthly.dw_Tmax = " << wGen->monthly.dw_Tmax[m] << "\n";
                stream << "wGen->monthly.monthlyTminDry = " << wGen->monthly.monthlyTminDry[m] << "\n";
                stream << "wGen->monthly.monthlyTmaxDry = " << wGen->monthly.monthlyTmaxDry[m] << "\n";
                stream << "wGen->monthly.monthlyTminWet = " << wGen->monthly.monthlyTminWet[m] << "\n";
                stream << "wGen->monthly.monthlyTmaxWet = " << wGen->monthly.monthlyTmaxWet[m] << "\n";
                stream << "wGen->monthly.stdDevTminDry = " << wGen->monthly.stDevTminDry[m] << "\n";
                stream << "wGen->monthly.stdDevTmaxDry = " << wGen->monthly.stDevTmaxDry[m] << "\n";
                stream << "wGen->monthly.stdDevTminWet = " << wGen->monthly.stDevTminWet[m] << "\n";
                stream << "wGen->monthly.stdDevTmaxWet = " << wGen->monthly.stDevTmaxWet[m] << "\n";

                stream << "-------------------------------------------" << "\n";
            }
        }
        else
        {
            cout << "...Write WG climate file -->" << outputFileName.toStdString() << "\n";

            QFile file(outputFileName);
            file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);

            QTextStream stream( &file );
            for (m=0; m<12; m++)
            {
                stream << m +1 << "\n";
                stream <<  wGen->monthly.monthlyTmin[m] << "\n";
                stream <<  wGen->monthly.monthlyTmax[m] << "\n";
                stream <<  wGen->monthly.sumPrec[m] << "\n";
                stream <<  wGen->monthly.stDevTmin[m] << "\n";
                stream <<  wGen->monthly.stDevTmax[m] << "\n";
                stream <<  wGen->monthly.fractionWetDays[m] << "\n";
                stream <<  wGen->monthly.probabilityWetWet[m] << "\n";
                stream <<  wGen->monthly.dw_Tmax[m] << "\n";
                stream <<  wGen->monthly.monthlyTminDry[m] << "\n";
                stream <<  wGen->monthly.monthlyTmaxDry[m] << "\n";
                stream <<  wGen->monthly.monthlyTminWet[m] << "\n";
                stream <<  wGen->monthly.monthlyTmaxWet[m] << "\n";
                stream << wGen->monthly.stDevTminDry[m] << "\n";
                stream << wGen->monthly.stDevTmaxDry[m] << "\n";
                stream << wGen->monthly.stDevTminWet[m] << "\n";
                stream << wGen->monthly.stDevTmaxWet[m] << "\n";
                for (int iBin=0;iBin<nrConsecutiveDryDaysBins;iBin++)
                {
                    stream << consecutiveDry[m][iBin] << "\t" ;
                }
                stream << "\n";
                for (int iBin=0;iBin<nrConsecutiveDryDaysBins;iBin++)
                {
                    stream << consecutiveWet[m][iBin] << "\t" ;
                }
                stream << "\n";
            }
        }
    }

    return true;
}

bool computeWG2DClimate(int nrDays, Crit3DDate inputFirstDate, float *inputTMin, float *inputTMax,
                      float *inputPrec, float precThreshold, float minPrecData,
                      TweatherGenClimate* wGen, bool writeOutput,bool outputForStats, QString outputFileName,
                      float* monthlyPrecipitation)
{
    long nValidData = 0;
    float dataPresence = 0;
    double sumTMin[12] = {0};
    double sumTMax[12] = {0};
    double sumPrec[12] = {0};
    double sumTMin2[12] = {0};
    double sumTMax2[12] = {0};
    long nWetDays[12] = {0};
    long nWetWetDays[12] = {0};
    long nDryDays[12] = {0};
    long nrData[12] = {0};
    double sumTmaxWet[12] = {0};
    double sumTmaxDry[12] = {0};
    double sumTminWet[12] = {0};
    double sumTminDry[12] = {0};
    double sumTmaxWet2[12] = {0};
    double sumTmaxDry2[12] = {0};
    double sumTminWet2[12] = {0};
    double sumTminDry2[12] = {0};
    int daysInMonth;
    bool isPreviousDayWet = false;

    // read data
    int m;
    Crit3DDate myDate = inputFirstDate;
    for (int n = 0; n < nrDays; n++)
    {
        m = myDate.month - 1;

        // the day is valid if all values are different from nodata
        if (int(inputTMin[n]) != int(NODATA)
            && int(inputTMax[n]) != int(NODATA)
            && int(inputPrec[n]) != int(NODATA))
        {
            nValidData++;
            nrData[m]++;
            sumTMin[m] += double(inputTMin[n]);
            sumTMin2[m] += double(inputTMin[n] * inputTMin[n]);
            sumTMax[m] += double(inputTMax[n]);
            sumTMax2[m] += double(inputTMax[n] * inputTMax[n]);
            sumPrec[m] += double(inputPrec[n]);

            if (inputPrec[n] > precThreshold)
            {
                if (isPreviousDayWet) nWetWetDays[m]++;
                nWetDays[m]++;
                sumTmaxWet[m] += double(inputTMax[n]);
                sumTminWet[m] += double(inputTMin[n]);
                sumTmaxWet2[m] += double(inputTMax[n] * inputTMax[n]);
                sumTminWet2[m] += double(inputTMin[n] * inputTMin[n]);
                isPreviousDayWet = true;
            }
            else
            {
                nDryDays[m]++;
                sumTmaxDry[m] += double(inputTMax[n]);
                sumTminDry[m] += double(inputTMin[n]);
                sumTmaxDry2[m] += double(inputTMax[n] * inputTMax[n]);
                sumTminDry2[m] += double(inputTMin[n] * inputTMin[n]);
                isPreviousDayWet = false;
            }
        }

        ++myDate;
    }

    dataPresence = float(nValidData) / float(nrDays);
    if (dataPresence < minPrecData)
        return false;

    // compute Climate
    for (m=0; m<12; m++)
    {
        if (nrData[m] > 0)
        {
            wGen->monthly.monthlyTmax[m] = sumTMax[m] / nrData[m]; //computes mean monthly values of maximum temperature
            wGen->monthly.monthlyTmin[m] = sumTMin[m] / nrData[m]; //computes mean monthly values of minimum temperature
            wGen->monthly.monthlyTmaxDry[m] = sumTmaxDry[m] / nDryDays[m];
            wGen->monthly.monthlyTmaxWet[m] = sumTmaxWet[m] / nWetDays[m];
            wGen->monthly.monthlyTminDry[m] = sumTminDry[m] / nDryDays[m];
            wGen->monthly.monthlyTminWet[m] = sumTminWet[m] / nWetDays[m];

            daysInMonth = getDaysInMonth(m+1,2001); // year = 2001 is to avoid leap year

            wGen->monthly.sumPrec[m] = sumPrec[m] / nrData[m] * daysInMonth;

            wGen->monthly.fractionWetDays[m] = float(nWetDays[m]) / float(nrData[m]);
            wGen->monthly.probabilityWetWet[m] = float(nWetWetDays[m]) / float(nWetDays[m]);

            if ( (nDryDays[m] > 0) && (nWetDays[m] > 0) )
                wGen->monthly.dw_Tmax[m] = (sumTmaxDry[m] / nDryDays[m]) - (sumTmaxWet[m] / nWetDays[m]);
            else
                wGen->monthly.dw_Tmax[m] = 0;

            wGen->monthly.stDevTmax[m] = sqrt(MAXVALUE(nrData[m]*sumTMax2[m]-(sumTMax[m]*sumTMax[m]), 0) / (nrData[m]*(nrData[m]-1)));
            wGen->monthly.stDevTmin[m] = sqrt(MAXVALUE(nrData[m]*sumTMin2[m]-(sumTMin[m]*sumTMin[m]), 0) / (nrData[m]*(nrData[m]-1)));
            wGen->monthly.stDevTmaxDry[m] = sqrt(MAXVALUE(nDryDays[m]*sumTmaxDry2[m]-(sumTmaxDry[m]*sumTmaxDry[m]), 0) / (nDryDays[m]*(nDryDays[m]-1)));
            wGen->monthly.stDevTmaxWet[m] = sqrt(MAXVALUE(nWetDays[m]*sumTmaxWet2[m]-(sumTmaxWet[m]*sumTmaxWet[m]), 0) / (nWetDays[m]*(nWetDays[m]-1)));
            wGen->monthly.stDevTminDry[m] = sqrt(MAXVALUE(nDryDays[m]*sumTminDry2[m]-(sumTminDry[m]*sumTminDry[m]), 0) / (nDryDays[m]*(nDryDays[m]-1)));
            wGen->monthly.stDevTminWet[m] = sqrt(MAXVALUE(nWetDays[m]*sumTminWet2[m]-(sumTminWet[m]*sumTminWet[m]), 0) / (nWetDays[m]*(nWetDays[m]-1)));

        }
        else
        {
            wGen->monthly.monthlyTmax[m] = NODATA;
            wGen->monthly.monthlyTmin[m] = NODATA;
            wGen->monthly.sumPrec[m] = NODATA;
            wGen->monthly.fractionWetDays[m] = NODATA;
            wGen->monthly.stDevTmax[m] = NODATA;
            wGen->monthly.stDevTmin[m] = NODATA;
            wGen->monthly.dw_Tmax[m] = NODATA;
            wGen->monthly.stDevTmaxDry[m] = NODATA;
            wGen->monthly.stDevTminDry[m] = NODATA;
            wGen->monthly.stDevTmaxWet[m] = NODATA;
            wGen->monthly.stDevTminWet[m] = NODATA;
        }
        monthlyPrecipitation[m] = wGen->monthly.sumPrec[m];
    }


    if (writeOutput)
    {
        if (!outputForStats)
        {
            cout << "...Write WG climate file -->" << outputFileName.toStdString() << "\n";

            QFile file(outputFileName);
            file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);

            QTextStream stream( &file );
            stream << "----------------- CLIMATE ----------------\n";
            for (m=0; m<12; m++)
            {
                stream << "month = " << m +1 << "\n";
                stream << "wGen->monthly.monthlyTmin = " << wGen->monthly.monthlyTmin[m] << "\n";
                stream << "wGen->monthly.monthlyTmax = " << wGen->monthly.monthlyTmax[m] << "\n";
                stream << "wGen->monthly.sumPrec = " << wGen->monthly.sumPrec[m] << "\n";
                stream << "wGen->monthly.stDevTmin = " << wGen->monthly.stDevTmin[m] << "\n";
                stream << "wGen->monthly.stDevTmax = " << wGen->monthly.stDevTmax[m] << "\n";
                stream << "wGen->monthly.fractionWetDays = " << wGen->monthly.fractionWetDays[m] << "\n";
                stream << "wGen->monthly.probabilityWetWet = " << wGen->monthly.probabilityWetWet[m] << "\n";
                stream << "wGen->monthly.dw_Tmax = " << wGen->monthly.dw_Tmax[m] << "\n";
                stream << "wGen->monthly.monthlyTminDry = " << wGen->monthly.monthlyTminDry[m] << "\n";
                stream << "wGen->monthly.monthlyTmaxDry = " << wGen->monthly.monthlyTmaxDry[m] << "\n";
                stream << "wGen->monthly.monthlyTminWet = " << wGen->monthly.monthlyTminWet[m] << "\n";
                stream << "wGen->monthly.monthlyTmaxWet = " << wGen->monthly.monthlyTmaxWet[m] << "\n";
                stream << "wGen->monthly.stdDevTminDry = " << wGen->monthly.stDevTminDry[m] << "\n";
                stream << "wGen->monthly.stdDevTmaxDry = " << wGen->monthly.stDevTmaxDry[m] << "\n";
                stream << "wGen->monthly.stdDevTminWet = " << wGen->monthly.stDevTminWet[m] << "\n";
                stream << "wGen->monthly.stdDevTmaxWet = " << wGen->monthly.stDevTmaxWet[m] << "\n";

                stream << "-------------------------------------------" << "\n";
            }
        }
        else
        {
            cout << "...Write WG climate file -->" << outputFileName.toStdString() << "\n";

            QFile file(outputFileName);
            file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);

            QTextStream stream( &file );
            for (m=0; m<12; m++)
            {
                stream << m +1 << "\n";
                stream <<  wGen->monthly.monthlyTmin[m] << "\n";
                stream <<  wGen->monthly.monthlyTmax[m] << "\n";
                stream <<  wGen->monthly.sumPrec[m] << "\n";
                stream <<  wGen->monthly.stDevTmin[m] << "\n";
                stream <<  wGen->monthly.stDevTmax[m] << "\n";
                stream <<  wGen->monthly.fractionWetDays[m] << "\n";
                stream <<  wGen->monthly.probabilityWetWet[m] << "\n";
                stream <<  wGen->monthly.dw_Tmax[m] << "\n";
                stream <<  wGen->monthly.monthlyTminDry[m] << "\n";
                stream <<  wGen->monthly.monthlyTmaxDry[m] << "\n";
                stream <<  wGen->monthly.monthlyTminWet[m] << "\n";
                stream <<  wGen->monthly.monthlyTmaxWet[m] << "\n";
                stream << wGen->monthly.stDevTminDry[m] << "\n";
                stream << wGen->monthly.stDevTmaxDry[m] << "\n";
                stream << wGen->monthly.stDevTminWet[m] << "\n";
                stream << wGen->monthly.stDevTmaxWet[m] << "\n";
            }
        }
    }

    return true;
}

/*!
  * \brief Generates a climate starting from daily weather
  */
bool climateGenerator(int nrData, TinputObsData climateDailyObsData, Crit3DDate climateDateIni,
                      Crit3DDate climateDateFin, float precThreshold, float minPrecData, TweatherGenClimate* wGen)
{
    unsigned int nrDays;
    int startIndex;
    TinputObsData newDailyObsData;
    bool result = false;

    startIndex = difference(climateDailyObsData.inputFirstDate, climateDateIni);  // starts from 0
    nrDays = difference(climateDateIni, climateDateFin)+1;

    newDailyObsData.inputFirstDate = climateDateIni;
    newDailyObsData.inputTMin.resize(nrDays);
    newDailyObsData.inputTMax.resize(nrDays);
    newDailyObsData.inputPrecip.resize(nrDays);

    int j = 0;

    for (int i = 0; i < nrData; i++)
    {
        if (i >= startIndex && i < (startIndex+int(nrDays)))
        {
            newDailyObsData.inputTMin[j] = climateDailyObsData.inputTMin[i];
            newDailyObsData.inputTMax[j] = climateDailyObsData.inputTMax[i];
            newDailyObsData.inputPrecip[j] = climateDailyObsData.inputPrecip[i];
            j++;
        }
    }

    result = computeWGClimate(nrDays, newDailyObsData.inputFirstDate, newDailyObsData.inputTMin,
                              newDailyObsData.inputTMax, newDailyObsData.inputPrecip,
                              precThreshold, minPrecData, wGen, false, "");

    newDailyObsData.inputTMin.clear();
    newDailyObsData.inputTMax.clear();
    newDailyObsData.inputPrecip.clear();

    return result;
}


/*!
  * \brief Compute sample standard deviation
*/
float sampleStdDeviation(float values[], int nElement)
{
    float sum = 0;
    float sum2 = 0;
    int i;

    float stdDeviation = 0;

    if (nElement <= 1)
        stdDeviation = NODATA;
    else
    {
        for (i = 0; i < nElement; i++)
        {
            sum = sum + values[i];
            sum2 = sum2 + (values[i] * values[i]);
        }

        stdDeviation = sqrt( std::max(nElement * sum2 - (sum * sum), 0.f) / float(nElement * (nElement - 1)) );
    }
    return stdDeviation;
}
