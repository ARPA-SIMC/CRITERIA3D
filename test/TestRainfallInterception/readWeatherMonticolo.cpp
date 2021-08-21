#include <stdlib.h>
#include <math.h>
#include <string.h>


#include "commonConstants.h"
#include "readWeatherMonticolo.h"



int readWeatherLineFileNumber(FILE *fp)
{
    int counter = -2;
    char dummy;

    do {
        dummy = getc(fp);
        if (dummy == '\n') counter++ ;
    } while (dummy != EOF);
    return counter ;
}

bool readWeatherMonticoloHalfhourlyData(FILE *fp,int* minute, int* hour, int *doy,int *day,int *month, int* year,double* temp,double* prec, int nrLines)
{
    char dummy ;
    int counter;
    char minutechar[5], hourchar[5], daychar[5],monthchar[5],yearchar[5],tempchar[20],precchar[20];
    // take out the first line (header)
    do {
        dummy = getc(fp);
    } while (dummy != '\n');

    // read data
    for (int iLines = 0 ; iLines < nrLines ; iLines++)
    {
        // initialize char
        for (int i=0; i<5 ; i++)
        {
            minutechar[i] = '\0';
            hourchar[i] = '\0';
            daychar[i] = '\0';
            monthchar[i] = '\0';
            yearchar[i] = '\0';
            tempchar[i] = '\0';
            precchar[i] = '\0';
            tempchar[i+5] = '\0';
            precchar[i+5] = '\0';
            tempchar[i+10] = '\0';
            precchar[i+10] = '\0';
            tempchar[i+15] = '\0';
            precchar[i+15] = '\0';
        }
        do {
            dummy = getc(fp);
        } while (dummy != ',');
        counter = 0;
        do {
            dummy = getc(fp);
            if (dummy != ',')
                tempchar[counter] = dummy;
            counter++;
        } while (dummy != ',');
        do {
            dummy = getc(fp);
        } while (dummy != ',');
        counter = 0;
        do {
            dummy = getc(fp);
            if (dummy != '/')
                daychar[counter] = dummy;
            counter++;
        } while (dummy != '/');
        counter = 0;
        do {
            dummy = getc(fp);
            if (dummy != '/')
                monthchar[counter] = dummy;
            counter++;
        } while (dummy != '/');
        counter = 0;
        yearchar[counter++] = getc(fp);
        yearchar[counter++] = getc(fp);
        yearchar[counter++] = getc(fp);
        yearchar[counter++] = getc(fp);
        counter = 0;
        getc(fp);
        do {
            dummy = getc(fp);
            if (dummy != ':')
                hourchar[counter] = dummy;
            counter++;
        } while (dummy != ':');
        counter = 0;
        do {
            dummy = getc(fp);
            if (dummy != ',')
                minutechar[counter] = dummy;
            counter++;
        } while (dummy != ',');
        counter = 0;
        do {
            dummy = getc(fp);
            if (dummy != '\n')
                precchar[counter] = dummy;
            counter++;
        } while (dummy != '\n');

        minute[iLines] = atoi(minutechar);
        hour[iLines] = atoi(hourchar);
        day[iLines] = atoi(daychar);
        month[iLines] = atoi(monthchar);
        year[iLines] = atoi(yearchar);
        temp[iLines] = atof(tempchar);
        prec[iLines] = atof(precchar);
        doy[iLines] = getDoyFromDate(day[iLines],month[iLines],year[iLines]);
        //printf("%.2d:%.2d, %.2d/%.2d/%.4d, %.2f , %.2f\n",hour[iLines],minute[iLines],day[iLines],month[iLines],year[iLines],temp[iLines],prec[iLines]);
        //getchar();
    }

}

int getDoyFromDate(int day, int month, int year)
{
        char leap = 0 ;
        int doy;
        int monthList[12];


        if (year%4 == 0)
        {
            leap = 1;
            if (year%100 == 0 && year%400 != 0) leap = 0 ;
        }
        monthList[0]=31 ;
        monthList[1]=28 + leap ;
        monthList[2]=31 ;
        monthList[3]=30 ;
        monthList[4]=31 ;
        monthList[5]=30 ;
        monthList[6]=31 ;
        monthList[7]=31 ;
        monthList[8]=30 ;
        monthList[9]=31 ;
        monthList[10]=30;
        monthList[11]=31 ;
        if (month < 1 || month > 12) return(PARAMETER_ERROR);
        if (day < 1 || day > monthList[month-1]) return(PARAMETER_ERROR);
        doy = 0 ;
        for (short i = 0 ; i < month-1 ; i++){
            doy += monthList[i] ;
        }
        doy += day ;
        return doy ;
}


bool getDateFromDoy(int doy,int year,int* month, int* day)
{
    char leap = 0 ;
    int monthList[12];


    if (year%4 == 0)
    {
        leap = 1;
        if (year%100 == 0 && year%400 != 0) leap = 0 ;
    }
    monthList[0]= 31;
    monthList[1]= monthList[0] + 28 + leap;
    monthList[2]= monthList[1] + 31 ;
    monthList[3]= monthList[2] + 30 ;
    monthList[4]= monthList[3] + 31 ;
    monthList[5]= monthList[4] + 30 ;
    monthList[6]= monthList[5] + 31 ;
    monthList[7]= monthList[6] + 31 ;
    monthList[8]= monthList[7] + 30 ;
    monthList[9]= monthList[8] + 31 ;
    monthList[10]= monthList[9] + 30;
    monthList[11]= monthList[10] + 31;
    if (doy > monthList[11])
    {
        return false;
    }
    else if (doy > monthList[10])
    {
        *month = 12;
        *day = doy - monthList[10];
    }
    else if (doy > monthList[9])
    {
        *month = 11;
        *day = doy - monthList[9];
    }
    else if (doy > monthList[8])
    {
        *month = 10;
        *day = doy - monthList[8];
    }
    else if (doy > monthList[7])
    {
        *month = 9;
        *day = doy - monthList[7];
    }
    else if (doy > monthList[6])
    {
        *month = 8;
        *day = doy - monthList[6];
    }
    else if (doy > monthList[5])
    {
        *month = 7;
        *day = doy - monthList[5];
    }
    else if (doy > monthList[4])
    {
        *month = 6;
        *day = doy - monthList[4];
    }
    else if (doy > monthList[3])
    {
        *month = 5;
        *day = doy - monthList[3];
    }
    else if (doy > monthList[2])
    {
        *month = 4;
        *day = doy - monthList[2];
    }
    else if (doy > monthList[1])
    {
        *month = 3;
        *day = doy - monthList[1];
    }
    else if (doy > monthList[0])
    {
        *month = 2;
        *day = doy - monthList[0];
    }
    else
    {
        *day = doy;
    }
    return true;
}
