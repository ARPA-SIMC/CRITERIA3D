/*!
    CRITERIA3D

    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

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

    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/

#include <math.h>
#include <stdio.h>

#include "crit3dDate.h"


#ifndef NODATA
    #define NODATA -9999
#endif

const long daysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};


// index: 1 - 12
int getDaysInMonth(int month, int year)
{
    if (month < 1 || month > 12) return NODATA;

    if(month == 2 && isLeapYear(year)) return 29;

    else return daysInMonth[month-1];
}


Crit3DDate::Crit3DDate()
{
    day = 0; month = 0; year = 0;
}


Crit3DDate::Crit3DDate(int myDay, int myMonth, int myYear)
{
    day = myDay; month = myMonth; year = myYear;
}


// myDate have to be in ISO 8601 form (YYYY-MM-DD)
Crit3DDate::Crit3DDate(std::string myDate)
{
    sscanf(myDate.data(), "%d-%02d-%02d", &year, &month, &day);
}


void Crit3DDate::setDate(int myDay, int myMonth, int myYear)
{
    day = myDay; month = myMonth; year = myYear;
}


bool operator == (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day == myDate2.day );
}


bool operator != (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return ! ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day == myDate2.day );
}


bool operator > (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return  ( myDate1.year > myDate2.year ||
            ( myDate1.year == myDate2.year && myDate1.month > myDate2.month ) ||
            ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day > myDate2.day ));
}


bool operator >= (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return ( myDate1.year > myDate2.year ||
           ( myDate1.year == myDate2.year && myDate1.month > myDate2.month ) ||
           ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day >= myDate2.day ));
}


bool operator < (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return  ( myDate1.year < myDate2.year ||
            ( myDate1.year == myDate2.year && myDate1.month < myDate2.month ) ||
            ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day < myDate2.day ));
}


bool operator <= (const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    return  ( myDate1.year < myDate2.year ||
            ( myDate1.year == myDate2.year && myDate1.month < myDate2.month ) ||
            ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day <= myDate2.day ));
}


Crit3DDate& operator ++ (Crit3DDate& myDate)
{
    if (myDate.day < getDaysInMonth(myDate.month, myDate.year))
    {
        myDate.day++;
    }
    else
    {
        if (myDate.month < 12)
        {
            myDate.month++;
        }
        else
        {
            myDate.year++;
            myDate.month = 1;
        }
        myDate.day = 1;
    }

    return myDate;
}


Crit3DDate& operator -- (Crit3DDate& myDate)
{
    if (myDate.day > 1)
    {
        myDate.day--;
    }
    else
    {
        if (myDate.month > 1)
        {
            myDate.month--;
        }
        else
        {
            myDate.year--;
            myDate.month = 12;
        }
        myDate.day = getDaysInMonth(myDate.month, myDate.year);
    }

    return myDate;
}


Crit3DDate Crit3DDate::addDays(long nrDays) const
{
    Crit3DDate myDate = *this;

    if (nrDays >= 0)
    {
        while (nrDays > 365)
        {
            int currentDoy = getDoyFromDate(myDate);
            int endYearDoy = getDoyFromDate(Crit3DDate(31, 12, myDate.year));
            nrDays -= (endYearDoy - currentDoy + 1);
            myDate.setDate(1, 1, myDate.year + 1);
        }
        for (int i = 0; i < nrDays; i++)
            ++myDate;
    }
    else
    {
        while (abs(nrDays) > 365)
        {
            nrDays += getDoyFromDate(myDate);
            myDate.setDate(31, 12, myDate.year - 1);
        }
        for (int i = 0; i > nrDays; i--)
            --myDate;
    }

    return myDate;
}


int Crit3DDate::daysTo(const Crit3DDate& myDate) const
{
    Crit3DDate first = min(*this, myDate);
    Crit3DDate last = max(*this, myDate);

    int delta = 0;
    while (first.year < last.year)
    {
        int currentDoy = getDoyFromDate(first);
        int endYearDoy = getDoyFromDate(Crit3DDate(31, 12, first.year));
        delta += (endYearDoy - currentDoy + 1);
        first.setDate(1, 1, first.year + 1);
    }
    while (first < last)
    {
        delta++;
        ++first;
    }

    if (last == myDate)
        return delta;
    else
        return -delta;
}


Crit3DDate max(const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    if  ( myDate1.year > myDate2.year ||
        ( myDate1.year == myDate2.year && myDate1.month > myDate2.month ) ||
        ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day > myDate2.day ))
        return myDate1;
    else
        return myDate2;
}


Crit3DDate min(const Crit3DDate& myDate1, const Crit3DDate& myDate2)
{
    if  ( myDate1.year < myDate2.year ||
        ( myDate1.year == myDate2.year && myDate1.month < myDate2.month ) ||
        ( myDate1.year == myDate2.year && myDate1.month == myDate2.month && myDate1.day < myDate2.day ))
        return myDate1;
    else
        return myDate2;
}


Crit3DDate getDateFromDoy(int myYear, int myDoy)
{
    if (myDoy > 366)
    {
        return NO_DATE;
    }
    if (myDoy == 366 && isLeapYear(myYear) == false)
    {
        return NO_DATE;
    }
    Crit3DDate myDate(1, 1, myYear);
    myDate = myDate.addDays(myDoy - 1);

    return myDate;
}

Crit3DDate getNullDate()
{
    Crit3DDate* myDate = new Crit3DDate();
    return *myDate;
}

bool isNullDate(Crit3DDate myDate)
{
    return (myDate.day == 0 && myDate.month == 0 && myDate.year == 0);
}


int difference(Crit3DDate firstDate, Crit3DDate lastDate)
{
    int delta = 0;

    while (firstDate.year < lastDate.year)
    {
        int currentDoy = getDoyFromDate(firstDate);
        int endYearDoy = getDoyFromDate(Crit3DDate(31, 12, firstDate.year));
        delta += (endYearDoy - currentDoy + 1);
        firstDate.setDate(1, 1, firstDate.year + 1);
    }
    while (firstDate < lastDate)
    {
        delta++;
        ++firstDate;
    }

    return delta;
}


bool isLeapYear(int year)
{
    bool isLeap = false ;
    if (year%4 == 0)
    {
      isLeap = true;
      if (year%100 == 0)
          if (! (year%400 == 0)) isLeap = false;
    }
    return isLeap ;
}


int getDoyFromDate(const Crit3DDate& myDate)
{
    int myDoy = 0;
    for(int month = 1; month < myDate.month; month++)
    {
        myDoy += getDaysInMonth(month, myDate.year);
    }

    myDoy += myDate.day;

    return myDoy;
}


std::string Crit3DDate::toStdString()
{
    char myStr[11];
    sprintf (myStr, "%d-%02d-%02d", this->year, this->month, this->day);

    return std::string(myStr);
}


std::string Crit3DDate::toString()
{
    char myStr[9];
    sprintf (myStr, "%d%02d%02d", this->year, this->month, this->day);

    return std::string(myStr);
}

