#include <QCoreApplication>
#include <stdio.h>
#include <stdlib.h>

#include "rainfallInterception.h"
#include "readWeatherMonticolo.h"
#include "commonConstants.h"

double laiEstimation(int doy);
void getDailyTemperatures (double* meanT, double* maxT, double* minT, double* temp, int iLine,int nrLines, int* day);
double getEvapotranspiration (double meanT, double maxT, double minT, double timeLagInMinutes, double latitude, int doy);

int main(int argc, char *argv[])
{
    // read weather data
    int numberOfLines;
    FILE* fp;
    fp = fopen("weather_Monticolo_all_years_2019_04_21.csv","r");
    numberOfLines = readWeatherLineFileNumber(fp);
    fclose(fp);
    double* temp = (double*) calloc(numberOfLines,sizeof(double));
    double* prec = (double*) calloc(numberOfLines,sizeof(double));
    int* day = (int*) calloc(numberOfLines,sizeof(int));
    int* month = (int*) calloc(numberOfLines,sizeof(int));
    int* year = (int*) calloc(numberOfLines,sizeof(int));
    int* dayOfYear = (int*) calloc(numberOfLines,sizeof(int));
    int* hour = (int*) calloc(numberOfLines,sizeof(int));
    int* minute = (int*) calloc(numberOfLines,sizeof(int));
    FILE* fp1;
    fp1 = fopen("weather_Monticolo_all_years_2019_04_21.csv","r");
    readWeatherMonticoloHalfhourlyData(fp1,minute,hour,dayOfYear,day,month,year,temp, prec, numberOfLines);
    fclose(fp1);

    double minT,maxT,meanT;
    minT = maxT = meanT = NODATA;
    double storedWater = 0.0;
    FILE* fp2;
    fp2 = fopen("resultsMonticolo.csv","w");
    for (int iLine=0;iLine<numberOfLines;iLine++)
    {
        //QCoreApplication a(argc, argv);

        double waterFreeEvaporation = 0.01;
        double lai = -9999;
        double laiMin = 0.5;
        double lightExtinctionCoefficient = 0.6;
        double leafStorage = 0.3;
        double stemStorage = 0.105;
        double maxStemFlowRate = 0.15;
        double freeRainfall = 0;
        double soilWater = 0;
        double stemFlow = 0;
        double throughfall = 0;
        double drainage = 0;
        double latitude = 46.42;
        lai = laiEstimation(dayOfYear[iLine]);
        getDailyTemperatures(&meanT,&maxT, &minT, temp, iLine, numberOfLines, day);
        printf("%d   %.1f, %.1f, %.1f, %.1f\n",dayOfYear[iLine], minT,meanT,maxT,lai);
        double dailyTimeLag;
        if (hour[iLine+1] >= hour[iLine])
            dailyTimeLag = hour[iLine+1]*60 + minute[iLine+1] - hour[iLine]*60 - minute[iLine];
        else
            dailyTimeLag = (hour[iLine+1]+24)*60 + minute[iLine+1] - hour[iLine]*60 - minute[iLine];
        double ETP;
        ETP = getEvapotranspiration(meanT,maxT,minT,dailyTimeLag,latitude,dayOfYear[iLine]);

        canopy::waterManagementCanopy(&storedWater,prec[iLine],0.7*ETP,lai,laiMin,lightExtinctionCoefficient,leafStorage, stemStorage,maxStemFlowRate,&freeRainfall,&drainage,&stemFlow,&throughfall,&soilWater);
        //canopy::waterManagementCanopy(&storedWater,prec[iLine],waterFreeEvaporation,lai,laiMin,lightExtinctionCoefficient,leafStorage, stemStorage,maxStemFlowRate,&soilWater);
        fprintf(fp2,"%.2d,%.2d,%.2d,%.2d,%.4d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",hour[iLine],minute[iLine],day[iLine],month[iLine],year[iLine],temp[iLine],prec[iLine],soilWater,storedWater,freeRainfall,throughfall,stemFlow);
        //printf("soilWater %f storedWater%f",soilWater, storedWater);
        //printf("%f\n",canopy::canopyNoInterceptedRainfallHydrall(lai,0.,rainfall));
        //getchar();
    }
    fclose(fp2);
    return 0;
}

double laiEstimation(int doy)
{
    double lai;
    if (doy <= 100)
    {
        lai = 0.1;
    }
    else if (doy >= 330)
    {
        lai = 0.1;
    }
    else if (doy >= 150 && doy <= 250)
    {
        lai = 3.6;
    }
    else
    {
        if (doy < 150)
        {
            lai = 0.1 + 0.07*(doy-100);
        }
        else
        {
            lai = 3.6 - 0.043*(doy-250);
        }
    }
    return lai;

}

void getDailyTemperatures (double* meanT, double* maxT, double* minT, double* temp, int iLine,int nrLines, int* day)
{
    //int counter = 0;
    int nrMax;
    int nrMin;
    *maxT = -9999;
    *minT = 9999;
    nrMin = MAXVALUE(0,iLine - 48);
    nrMax = MINVALUE(nrLines,iLine + 48);
    for (int i=nrMin ; i <  nrMax; i++)
    {
        if (day[i] == day [iLine])
        {
            *maxT = MAXVALUE(*maxT, temp[i]);
            *minT = MINVALUE(*minT, temp[i]);
        }
    }
    *meanT = 0.5*((*maxT) + (*minT));
}

double getEvapotranspiration (double meanT, double maxT, double minT, double timeLagInMinutes, double latitude, int doy)
{
    double evapotranspirationDaily;
    double deg2rad = 0.01745;
    double hour2Radians = 24. * 60 / PI;
    double solarConstant = 0.082;
    double hargreavesConstant = 0.17;
    double distanceEarthSun,solarDeclination,latitudeInRadians,sunset,extraterrestrialRadiation;
    distanceEarthSun = 1 + 0.033*cos(deg2rad*doy);
    solarDeclination = 0.409*sin(deg2rad*doy - 1.39);
    latitudeInRadians = latitude * deg2rad;
    sunset = acos(-tan(latitudeInRadians)*tan(solarDeclination));
    extraterrestrialRadiation = hour2Radians*solarConstant*distanceEarthSun;
    extraterrestrialRadiation *= (sunset*sin(latitudeInRadians)*sin(solarDeclination)+ cos(latitudeInRadians)*cos(solarDeclination)*sin(sunset));
    evapotranspirationDaily = 0.0135/2.45*hargreavesConstant*sqrt(maxT-minT)*(meanT+17.8)*extraterrestrialRadiation;
    return (evapotranspirationDaily / (24*60) * timeLagInMinutes);





}
