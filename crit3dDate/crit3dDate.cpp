/*!
    \name crit3dDate.cpp
    \copyright (C) 2016 Fausto Tomei, Gabriele Antolini, Antonio Volta,
                        Alberto Pistocchi, Marco Bittelli, Laura Costantini

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

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include <math.h>
#include <stdio.h>

#include "crit3dDate.h"


#ifndef NODATA
    #define NODATA -9999
#endif

const long daysInMonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const long doyMonth[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};


// index: 1 - 12
int getDaysInMonth(int month, int year)
{
    if ((month < 1) || (month > 12)) return NODATA;

    if(month == 2 && isLeapYear(year))
        return 29;
    else
        return daysInMonth[month-1];
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


Crit3DDate Crit3DDate::addDays(long offset) const
{
    if (offset == 0) return (*this);

    long julianDay = getJulianDay(day, month, year);
    return getDateFromJulianDay(julianDay + offset);
}


int Crit3DDate::daysTo(const Crit3DDate& myDate) const
{
    long j1 = getJulianDay(this->day, this->month, this->year);
    long j2 = getJulianDay(myDate.day, myDate.month, myDate.year);
    return j2-j1;
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


Crit3DDate getDateFromDoy(int year, int doy)
{
    if (doy < 1) return NO_DATE;
    short month;

    // before 29 february
    if (doy <= 59)
    {
        month = (doy <= 31) ? 1 : 2;
        return Crit3DDate(doy-doyMonth[month-1], month, year);
    }

    const short leap = isLeapYear(year) ? 1 : 0;
    if (doy > (365 + leap)) return NO_DATE;

    // 29 february
    if (doy == 60 && leap == 1)
        return Crit3DDate(29, 2, year);

    // after
    month = 3;
    while (month <= 12 && doy > (doyMonth[month]+leap))
        month++;

    return Crit3DDate(doy-(doyMonth[month-1]+leap), month, year);
}

void Crit3DDate::setNullDate()
{
    day = 0;
    month = 0;
    year = 0;
}

bool Crit3DDate::isNullDate()
{
    return (day == 0 && month == 0 && year == 0);
}


int difference(Crit3DDate firstDate, Crit3DDate lastDate)
{
    return firstDate.daysTo(lastDate);
}


bool isLeapYear(int year)
{
    // No year 0 in Gregorian calendar, so -1, -5, -9 etc are leap years
    if (year < 1)
        ++year;

    if (year % 4 != 0) return false;
    if (year % 100 != 0) return true;
    return (year % 400 == 0);
}


int getDoyFromDate(const Crit3DDate& myDate)
{
    int doy = doyMonth[myDate.month-1] + myDate.day;
    if (myDate.month > 2)
        if (isLeapYear(myDate.year))
            doy++;

    return doy;
}


static inline long floordiv(long a, long b)
{
    return (a - (a < 0 ? b - 1 : 0)) / b;
}

inline long getJulianDay(int day, int month, int year)
{
    // Adjust for no year 0
    if (year < 0)
        ++year;

    /*
     * Math from The Calendar FAQ at http://www.tondering.dk/claus/cal/julperiod.php
     * This formula is correct for all julian days, when using mathematical integer
     * division (round to negative infinity), not c++11 integer division (round to zero)
     */
    const long a = floordiv(14 - month, 12);
    const long y = year + 4800 - a;
    const int  m = month + 12 * a - 3;
    return day + floordiv(153 * m + 2, 5) + 365 * y + floordiv(y, 4) - floordiv(y, 100) + floordiv(y, 400) - 32045;
}


Crit3DDate getDateFromJulianDay(long julianDay)
{
    /*
     * Math from The Calendar FAQ at http://www.tondering.dk/claus/cal/julperiod.php
     * This formula is correct for all julian days, when using mathematical integer
     * division (round to negative infinity), not c++11 integer division (round to zero)
     */

    const long a = julianDay + 32044;
    const long b = floordiv(4 * a + 3, 146097);
    const int  c = a - floordiv(146097 * b, 4);
    const int  d = floordiv(4 * c + 3, 1461);
    const int  e = c - floordiv(1461 * d, 4);
    const int  m = floordiv(5 * e + 2, 153);
    const int  day = e - floordiv(153 * m + 2, 5) + 1;
    const int  month = m + 3 - 12 * floordiv(m, 10);
    int  year = 100 * b + d - 4800 + floordiv(m, 10);

    // Adjust for no year 0
    if (year <= 0)
        --year ;

    return { day, month, year };
}


std::string Crit3DDate::toStdString()
{
    char myStr[11];
    sprintf (myStr, "%d-%02d-%02d", this->year, this->month, this->day);

    return std::string(myStr);
}


std::string Crit3DDate::toStdString() const
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


std::string Crit3DDate::toString() const
{
    char myStr[9];
    sprintf (myStr, "%d%02d%02d", this->year, this->month, this->day);

    return std::string(myStr);
}

