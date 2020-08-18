#ifndef READWEATHERMONTICOLO_H
#define READWEATHERMONTICOLO_H
#include <stdio.h>


int readWeatherLineFileNumber(FILE *fp);
bool readWeatherMonticoloHalfhourlyData(FILE *fp, int* minute, int* hour, int *doy, int *day, int *month, int* year, double* temp, double* prec, int nrLines);
int getDoyFromDate(int day, int month, int year);
bool getDateFromDoy(int doy,int year,int* month, int* day);

#endif // READWEATHERMONTICOLO_H
