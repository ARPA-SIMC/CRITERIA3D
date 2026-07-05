/*!
    \name crit3dTime.cpp
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
#include "crit3dDate.h"


Crit3DTime::Crit3DTime()
{
    date.day = 0;
    date.month = 0;
    date.year = 0;
    time = 0;
}

Crit3DTime::Crit3DTime(const Crit3DDate &myDate, int mySeconds)
    : date{myDate}
{
    time = 0;
    *this = addSeconds(mySeconds);
}


bool Crit3DTime::setFromISOString(const std::string &dateTimeStr)
{
    if (dateTimeStr.size() < 16)
        return false;

    int hour= 0;
    int minutes = 0;
    int seconds = 0;
    int result;

    if (dateTimeStr.at(10) == 'T')
    {
        // MYSQL format: yyyy-MM-ddThh:mm:ss:00
        result = sscanf(dateTimeStr.data(), "%04d-%02d-%02dT%02d:%02d:%02d", &date.year, &date.month, &date.day, &hour, &minutes, &seconds);
    }
    else
    {
        // ISO 8601: yyyy-MM-dd hh:mm:ss
        result = sscanf(dateTimeStr.data(), "%04d-%02d-%02d %02d:%02d:%02d", &date.year, &date.month, &date.day, &hour, &minutes, &seconds);
    }
    if (result < 4)
        return false;

    time = hour * HOUR_SECONDS + minutes * 60 + seconds;

    return true;
}


int Crit3DTime::getHour() const
{
    return (time / HOUR_SECONDS);
}

int Crit3DTime::getNearestHour() const
{
    return int(round(time / HOUR_SECONDS));
}

int Crit3DTime::getMinutes() const
{
    return (time - getHour()*HOUR_SECONDS) / 60;
}

int Crit3DTime::getSeconds() const
{
    return (time - getHour()*HOUR_SECONDS - getMinutes()*60);
}

bool operator < (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return (time1.date < time2.date ||
           (time1.date == time2.date && time1.time < time2.time));
}

bool operator > (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return (time1.date > time2.date ||
           (time1.date == time2.date && time1.time > time2.time));
}

bool operator <= (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return (time1 < time2 || time1.isEqual(time2));
}

bool operator >= (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return (time1 > time2 || time1.isEqual(time2));
}

bool operator == (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return time1.isEqual(time2);
}

bool operator != (const Crit3DTime& time1, const Crit3DTime& time2)
{
    return ! time1.isEqual(time2);
}

void Crit3DTime::setNullTime()
{
    date.setNullDate();
    time = 0;
}

bool Crit3DTime::isNullTime()
{
    return date.isNullDate() && time == 0;
}

bool Crit3DTime::isEqual(const Crit3DTime& myTime) const
{
    return (date == myTime.date && time == myTime.time);
}


Crit3DTime Crit3DTime::addSeconds(long mySeconds) const
{
    Crit3DTime myTime = *this;
    myTime.time += mySeconds;

    while (!((myTime.time >= 0) && (myTime.time < DAY_SECONDS)))
    {
        if (myTime.time >= DAY_SECONDS)
        {
            ++(myTime.date);
            myTime.time -= DAY_SECONDS;
        }
        else if (myTime.time < 0)
        {
            --(myTime.date);
            myTime.time += DAY_SECONDS;
        }
    }

    return myTime;
}


int Crit3DTime::hourTo(const Crit3DTime &newTime)
{
    int nrDays = this->date.daysTo(newTime.date);
    int nrHours = newTime.getHour() - this->getHour();
    return nrDays * 24 + nrHours;
}


std::string Crit3DTime::toISOString() const
{
    char myStr[17];
    sprintf (myStr, "%d-%02d-%02d %02d:%02d", this->date.year, this->date.month, this->date.day, this->getHour(), this->getMinutes());

    return std::string(myStr);
}


std::string Crit3DTime::toString() const
{
    char myStr[13];
    sprintf (myStr, "%d%02d%02dT%02d%02d", this->date.year, this->date.month, this->date.day, this->getHour(), this->getMinutes());

    return std::string(myStr);
}
