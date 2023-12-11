/*!
    \copyright 2020 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of AGROLIB.
    AGROLIB has been developed under contract issued by ARPAE Emilia-Romagna

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

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include <math.h>
#include <netcdf.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "netcdfHandler.h"
#include "crit3dDate.h"


using namespace std;


string lowerCase(string myStr)
{
    transform(myStr.begin(), myStr.end(), myStr.begin(), ::tolower);
    return myStr;
}


NetCDFVariable::NetCDFVariable()
    : name{""}, longName{""}, unit{""}, id{NODATA}, type{NODATA}
{ }


NetCDFVariable::NetCDFVariable(char*_name, int _id, int _type)
    : name{_name}, longName{_name}, unit(""), id{_id}, type{_type}
{ }


std::string NetCDFVariable::getVarName()
{
    std::string unitStr = "";

    if (unit != "")
    {
        unitStr = " - " + unit;
    }

    if (longName.size() <= 32 && longName != "")
    {
        return longName + unitStr;
    }
    else
    {
        return name + unitStr;
    }
}


NetCDFHandler::NetCDFHandler()
{
    x = nullptr;
    y = nullptr;
    lat = nullptr;
    lon = nullptr;
    time = nullptr;

    this->clear();
}


void NetCDFHandler::close()
{
    if (ncId != NODATA)
    {
        // Close file
        nc_close(ncId);
        ncId = NODATA;
    }

    this->clear();
}


void NetCDFHandler::clear()
{
    if (x != nullptr) delete [] x;
    if (y != nullptr) delete [] y;
    if (lat != nullptr) delete [] lat;
    if (lon != nullptr) delete [] lon;
    if (time != nullptr) delete [] time;

    utmZone = NODATA;

    ncId = NODATA;
    nrX = NODATA;
    nrY = NODATA;
    nrLat = NODATA;
    nrLon = NODATA;
    nrTime = NODATA;

    idX = NODATA;
    idY = NODATA;
    idLat = NODATA;
    idLon = NODATA;
    idTime = NODATA;
    idTimeBnds = NODATA;

    isUTM = false;
    isLatLon = false;
    isRotatedLatLon = false;
    isYincreasing = false;

    isStandardTime = false;
    isHourly = false;
    isDaily = false;

    firstDate = NO_DATE;

    x = nullptr;
    y = nullptr;
    lon = nullptr;
    lat = nullptr;
    time = nullptr;

    dataGrid.clear();
    dimensions.clear();
    variables.clear();
    metadata.str("");
    timeUnit = "";

    missingValue = NODATA;
}


int NetCDFHandler::getDimensionIndex(char* dimName)
{
    for (unsigned int i = 0; i < dimensions.size(); i++)
    {
        if (dimensions[i].name == std::string(dimName))
            return int(i);
    }

    return NODATA;
}


NetCDFVariable NetCDFHandler::getVariableFromId(int idVar)
{
    for (unsigned int i = 0; i < variables.size(); i++)
        if (variables[i].id == idVar)
            return variables[i];

    return NetCDFVariable();
}


NetCDFVariable NetCDFHandler::getVariableFromIndex(int index)
{
    if (unsigned(index) < variables.size())
    {
        return variables[index];
    }
    else
    {
        return NetCDFVariable();
    }
}


bool NetCDFHandler::setVarLongName(const std::string& varName, const string &varLongName)
{
    for (unsigned int i = 0; i < variables.size(); i++)
    {
        if (variables[i].name == varName)
        {
            variables[i].longName = varLongName;
            return true;
        }
    }

    return false;
}


bool NetCDFHandler::setVarUnit(const std::string& varName, const std::string &varUnit)
{
    for (unsigned int i = 0; i < variables.size(); i++)
    {
        if (variables[i].name == varName)
        {
            variables[i].unit = varUnit;
            return true;
        }
    }

    return false;
}


bool NetCDFHandler::isPointInside(gis::Crit3DGeoPoint geoPoint)
{
    if (isLatLon)
    {
        return geoPoint.isInsideGrid(latLonHeader);
    }
    else
    {
        gis::Crit3DUtmPoint utmPoint;
        gis::getUtmFromLatLon(utmZone, geoPoint, &utmPoint);
        return utmPoint.isInsideGrid(*(dataGrid.header));
    }
}


std::string NetCDFHandler::getDateTimeStr(int timeIndex)
{
    if (idTime == NODATA)
    {
        return "time dimension is not defined.";
    }

    if (timeIndex < 0 || timeIndex >= nrTime)
    {
        return "ERROR: time index is out of range.";
    }

    Crit3DTime myTime = getTime(timeIndex);
    if (myTime == NO_DATETIME)
    {
        return "ERROR: time is not standard (std: seconds since 1970-01-01)";
    }

    return myTime.toISOString();
}


Crit3DTime NetCDFHandler::getTime(int timeIndex)
{
    if (timeIndex < 0 || timeIndex >= nrTime)
    {
        return NO_DATETIME;
    }

    long nrDays, residualTime;

    if (isStandardTime)
    {
        long nrSeconds = long(time[timeIndex]);
        nrDays = int(floor(nrSeconds / DAY_SECONDS));
        residualTime = nrSeconds - (nrDays * long(DAY_SECONDS));
    }
    else if (isHourly)
    {
        long nrHours = long(time[timeIndex]);
        nrDays = int(floor(nrHours / 24));
        residualTime = (nrHours - nrDays*24) * long(HOUR_SECONDS);
    }
    else if (isDaily)
    {
        nrDays = int(time[timeIndex]);
        residualTime = 0;
    }
    else
    {
        return NO_DATETIME;
    }

    Crit3DDate myDate = Crit3DTime(firstDate, 0).date.addDays(nrDays);

    return Crit3DTime(myDate, residualTime);
}


bool NetCDFHandler::readProperties(string fileName)
{
    int retval;

    char name[NC_MAX_NAME+1];
    char attrName[NC_MAX_NAME+1];
    char varName[NC_MAX_NAME+1];
    char typeName[NC_MAX_NAME+1];

    int valueInt;
    double value;
    size_t length;
    nc_type ncTypeId;
    int timeType = NC_DOUBLE;

    metadata.str("");

    // NC_NOWRITE tells netCDF we want read-only access
    if ((retval = nc_open(fileName.data(), NC_NOWRITE, &ncId)))
    {
        metadata << nc_strerror(retval) << endl;
        return false;
    }

    // NC_INQ tells how many netCDF dimensions, variables and global attributes are in
    // the file, also the dimension id of the unlimited dimension, if there is one.
    int nrDimensions, nrVariables, nrGlobalAttributes, unlimDimensionId;
    if ((retval = nc_inq(ncId, &nrDimensions, &nrVariables, &nrGlobalAttributes, &unlimDimensionId)))
    {
        metadata << nc_strerror(retval) << endl;
        return false;
    }

    // GLOBAL ATTRIBUTES
    metadata << "FILE:" << endl << fileName << endl << endl;
    metadata << "Global attributes:" << endl;

    for (int a = 0; a < nrGlobalAttributes; a++)
    {
        nc_inq_attname(ncId, NC_GLOBAL, a, attrName);
        nc_inq_attlen(ncId, NC_GLOBAL, attrName, &length);

        char* valueChar = new char[length+1];
        nc_get_att_text(ncId, NC_GLOBAL, attrName, valueChar);

        string valueString = string(valueChar).substr(0, length);
        metadata << attrName << " = " << valueString << endl;

        delete [] valueChar;
   }

   // DIMENSIONS
   int varDimIds[NC_MAX_VAR_DIMS];
   int nrVarDimensions, nrVarAttributes;

   metadata << "\nDimensions: " << endl;
   for (int i = 0; i < nrDimensions; i++)
   {
       nc_inq_dim(ncId, i, name, &length);

       dimensions.push_back(NetCDFVariable(name, i, NODATA));

       if (lowerCase(string(name)) == "time")
       {
           nrTime = int(length);
       }
       else if (lowerCase(string(name)) == "x")
       {
           nrX = int(length);
           isUTM = true;
       }
       else if (lowerCase(string(name)) == "y")
       {
           nrY = int(length);
           isUTM = true;
       }
       else if (lowerCase(string(name)) == "lat" || lowerCase(string(name)) == "latitude")
       {
           nrLat = int(length);
           isLatLon = true;
       }
       else if (lowerCase(string(name)) == "lon" || lowerCase(string(name)) == "longitude")
       {
           nrLon = int(length);
           isLatLon = true;
       }
       else if (lowerCase(string(name)) == "rlat")
       {
           nrLat = int(length);
           isRotatedLatLon = true;
       }
       else if (lowerCase(string(name)) == "rlon")
       {
           nrLon = int(length);
           isRotatedLatLon = true;
       }

       metadata << i << " - " << name << "\t values: " << length << endl;
   }

   if (isLatLon)
   {
       metadata <<"\nLat Lon grid";
       metadata <<"\n(lon,lat) = "<< nrLon << "," <<nrLat << endl;
   }
   else if (isRotatedLatLon)
   {
       metadata <<"\nRotated pole grid";
       metadata <<"\n(rlon, rlat) = "<< nrLon << "," <<nrLat << endl;
   }
   else if (isUTM)
   {
       metadata <<"\nUTM grid";
       metadata <<"\n(x,y) = "<< nrX << "," << nrY << endl;
   }
   else
   {
       metadata <<"\n WARNING: missing spatial dimension!";
   }

   // VARIABLES
   metadata << "\nVariables: " << endl;
   for (int v = 0; v < nrVariables; v++)
   {
       nc_inq_var(ncId, v, varName, &ncTypeId, &nrVarDimensions, varDimIds, &nrVarAttributes);
       nc_inq_type(ncId, ncTypeId, typeName, &length);

       // is it a variable?
       if (nrVarDimensions > 1)
       {
            variables.push_back(NetCDFVariable(varName, v, ncTypeId));
       }
       else
       {
            int i = getDimensionIndex(varName);
            if (i != NODATA)
                dimensions[unsigned(i)].type = ncTypeId;
            else
                metadata << endl << "ERROR: dimension not found: " << varName << endl;
       }

       if (lowerCase(string(varName)) == "time")
       {
           idTime = v;
           nc_inq_vartype(ncId, v, &timeType);
       }
       if (isUTM)
       {
            if (lowerCase(string(varName)) == "x")
                idX = v;
            else if (lowerCase(string(varName)) == "y")
                idY = v;
       }
       if (isLatLon)
       {
           if (lowerCase(string(varName)) == "lat" || lowerCase(string(varName)) == "latitude")
               idLat = v;
           else if (lowerCase(string(varName)) == "lon" || lowerCase(string(varName)) == "longitude")
               idLon = v;
       }
       if (isRotatedLatLon)
       {
           if (lowerCase(string(varName)) == "rlat")
               idLat = v;
           else if (lowerCase(string(varName)) == "rlon")
               idLon = v;
       }

       metadata << endl << v  << "\t" << varName << "\t" << typeName << "\t dims: ";
       for (int d = 0; d < nrVarDimensions; d++)
       {
           nc_inq_dim(ncId, varDimIds[d], name, &length);
           metadata << name << " ";
       }
       metadata << endl;

       // ATTRIBUTES
       for (int a = 0; a < nrVarAttributes; a++)
       {
            nc_inq_attname(ncId, v, a, attrName);
            nc_inq_atttype(ncId, v, attrName, &ncTypeId);

            if (ncTypeId == NC_CHAR)
            {
                nc_inq_attlen(ncId, v, attrName, &length);

                char* valueChar = new char[length+1];
                nc_get_att_text(ncId, v, attrName, valueChar);

                string valueStr = string(valueChar).substr(0, length);
                metadata << attrName << " = " << valueStr << endl;

                if (lowerCase(string(attrName)) == "units" || lowerCase(string(attrName)) == "unit")
                {
                    if (v == idTime)
                    {
                        timeUnit = valueStr;

                        if (lowerCase(valueStr).substr(0, 13) == "seconds since")
                        {
                            isStandardTime = true;
                            std::string dateStr = lowerCase(valueStr).substr(14, 23);
                            firstDate = Crit3DDate(dateStr);
                        }
                        else if (lowerCase(valueStr).substr(0, 11) == "hours since")
                        {
                            isHourly = true;
                            std::string dateStr = lowerCase(valueStr).substr(12, 21);
                            firstDate = Crit3DDate(dateStr);
                        }
                        else if (lowerCase(valueStr).substr(0, 10) == "days since")
                        {
                            isDaily = true;
                            std::string dateStr = lowerCase(valueStr).substr(11, 20);
                            firstDate = Crit3DDate(dateStr);
                        } 
                    }
                    else
                    {
                        // set variable unit
                        setVarUnit(string(varName), valueStr);
                    }
                }

                if (lowerCase(string(attrName)) == "long_name")
                {
                    setVarLongName(string(varName), valueStr);
                }

                delete [] valueChar;
            }
            else if (ncTypeId == NC_INT)
            {
                nc_get_att(ncId, v, attrName, &valueInt);

                // no data
                if (lowerCase(string(attrName)) == "missing_value" || lowerCase(string(attrName)) == "nodata")
                {
                    missingValue = double(valueInt);
                }

                metadata << attrName << " = " << valueInt << endl;
            }
            else if (ncTypeId == NC_DOUBLE)
            {
                nc_get_att(ncId, v, attrName, &value);

                // no data
                if (lowerCase(string(attrName)) == "missing_value" || lowerCase(string(attrName)) == "nodata")
                {
                    missingValue = value;
                }

                metadata << attrName << " = " << value << endl;
            }
        }
    }

    if (isLatLon || isRotatedLatLon)
    {
        if (idLat != NODATA && idLon != NODATA)
        {
            lat = new float[unsigned(nrLat)];
            lon = new float[unsigned(nrLon)];

            if ((retval = nc_get_var_float(ncId, idLon, lon)))
                metadata << "\nERROR in reading longitude:" << nc_strerror(retval);

            if ((retval = nc_get_var_float(ncId, idLat, lat)))
                metadata << "\nERROR in reading latitude:" << nc_strerror(retval);

            metadata << endl << "lat:" << endl;
            for (int i = 0; i < nrLat; i++)
                metadata << lat[i] << ", ";
            metadata << endl << "lon:" << endl;
            for (int i = 0; i < nrLon; i++)
                metadata << lon[i] << ", ";
            metadata << endl;

            latLonHeader.nrRows = nrLat;
            latLonHeader.nrCols = nrLon;

            latLonHeader.llCorner.longitude = double(lon[0]);
            latLonHeader.dx = double(lon[nrLon-1] - lon[0]) / double(nrLon-1);
            latLonHeader.dy = fabs(lat[nrLat-1] - lat[0]) / double(nrLat-1);

            isYincreasing = (lat[1] > lat[0]);
            if (isYincreasing)
            {
                latLonHeader.llCorner.latitude = double(lat[0]);
            }
            else
            {
                latLonHeader.llCorner.latitude = double(lat[nrLat-1]);
            }

            latLonHeader.llCorner.longitude -= (latLonHeader.dx * 0.5);
            latLonHeader.llCorner.latitude -= (latLonHeader.dy * 0.5);

            latLonHeader.flag = NODATA;

            // raster header
            dataGrid.header->nrRows = latLonHeader.nrRows;
            dataGrid.header->nrCols = latLonHeader.nrCols;
            dataGrid.header->flag = latLonHeader.flag;
            dataGrid.header->llCorner.y = latLonHeader.llCorner.latitude;
            dataGrid.header->llCorner.x = latLonHeader.llCorner.longitude;
            // avg value (not used)
            dataGrid.header->cellSize = (latLonHeader.dx + latLonHeader.dy) * 0.5;
            dataGrid.initializeGrid(0);
        }
    }
    else if (isUTM)
    {
        if (idX != NODATA && idY != NODATA)
        {
            x = new float[unsigned(nrX)];
            if ((retval = nc_get_var_float(ncId, idX, x)))
            {
                metadata << endl << "ERROR in reading x: " << nc_strerror(retval);
                nc_close(ncId);
                return false;
            }

            y = new float[unsigned(nrY)];
            if ((retval = nc_get_var_float(ncId, idY, y)))
            {
                metadata << endl << "ERROR in reading y: " << nc_strerror(retval);
                nc_close(ncId);
                return false;
            }

            if (! isEqual(x[1]-x[0], y[1]-y[0]))
                metadata << "\nWarning! dx != dy" << endl;

            dataGrid.header->cellSize = double(x[1]-x[0]);
            dataGrid.header->llCorner.x = double(x[0]);

            isYincreasing = (y[1] > y[0]);
            if (isYincreasing)
            {
                dataGrid.header->llCorner.y = double(y[0]);
            }
            else
            {
                dataGrid.header->llCorner.y = double(y[nrY-1]);
            }

            dataGrid.header->llCorner.x -= dataGrid.header->cellSize * 0.5;
            dataGrid.header->llCorner.y -= dataGrid.header->cellSize * 0.5;

            dataGrid.header->nrCols = nrX;
            dataGrid.header->nrRows = nrY;
            dataGrid.header->flag = NODATA;
            dataGrid.initializeGrid(0);
        }
        else
            metadata << endl << "ERROR: missing x,y data" << endl;
    }

    // TIME
    if (isStandardTime || isHourly || isDaily)
    {
        time = new double[unsigned(nrTime)];

        if (timeType == NC_DOUBLE)
        {
            retval = nc_get_var(ncId, idTime, time);
        }
        else if (timeType == NC_FLOAT)
        {
            float* floatTime = new float[unsigned(nrTime)];
            retval = nc_get_var_float(ncId, idTime, floatTime);
            for (int i = 0; i < nrTime; i++)
            {
                time[i] = double(floatTime[i]);
            }
            delete [] floatTime;
        }
        else if (timeType == NC_INT || timeType == NC_INT64)
        {
            int* intTime = new int[unsigned(nrTime)];
            retval = nc_get_var_int(ncId, idTime, intTime);
            for (int i = 0; i < nrTime; i++)
            {
                time[i] = double(intTime[i]);
            }
            delete [] intTime;
        }
        else
        {
            metadata << "Error! Not valid time data type: " << timeType << endl;
            return false;
        }
    }

    metadata << endl << "first date: " << getDateTimeStr(0) << endl;
    metadata << "last date: " << getDateTimeStr(nrTime-1) << endl;

    metadata << endl << "VARIABLES list:" << endl;
    for (unsigned int i = 0; i < variables.size(); i++)
    {
        metadata << variables[i].getVarName() << endl;
    }

   return true;
}


bool NetCDFHandler::exportDataSeries(int idVar, gis::Crit3DGeoPoint geoPoint, Crit3DTime seriesFirstTime, Crit3DTime seriesLastTime, stringstream *buffer)
{
    // check
    if (! isTimeReadable())
    {
        *buffer << "Wrong or missing time dimension!" << endl;
        return false;
    }
    if (! isPointInside(geoPoint))
    {
        *buffer << "Wrong position!" << endl;
        return false;
    }

    if (seriesFirstTime < getFirstTime() || seriesLastTime > getLastTime())
    {
        *buffer << "Wrong time index!" << endl;
        return false;
    }

    // find row, col
    int row, col;
    if (isLatLon)
    {
        gis::getRowColFromLatLon(latLonHeader, geoPoint, &row, &col);
        if (isYincreasing)
            row = (nrLat-1) - row;
    }
    else
    {
        gis::Crit3DUtmPoint utmPoint;
        gis::getUtmFromLatLon(utmZone, geoPoint, &utmPoint);
        gis::getRowColFromXY(*(dataGrid.header), utmPoint, &row, &col);
        row = (nrY -1) - row;
    }

    // find time indexes
    int t1 = NODATA;
    int t2 = NODATA;
    int i = 0;
    while ((i < nrTime) && (t1 == NODATA || t2 == NODATA))
    {
        if (getTime(i) == seriesFirstTime)
            t1 = i;
        if (getTime(i) == seriesLastTime)
            t2 = i;
        i++;
    }

    // check time
    if  (t1 == NODATA || t2 == NODATA)
    {
        *buffer << "Time out of range!" << endl;
        return false;
    }

    // check variable
    NetCDFVariable var = getVariableFromId(idVar);
    if (var.getVarName() == "")
    {
        *buffer << "Wrong variable!" << endl;
        return false;
    }

    *buffer << "variable: " << var.getVarName() << endl;

    // write position
    if (isLatLon)
     {
        *buffer << "lat: " << lat[row] << "\tlon: " << lon[col] << endl;
    }
    else
    {
        *buffer << "utm x: " << x[col] << "\tutm y: " << y[row] << endl;
    }

    *buffer << endl;

    // write data
    size_t* index = new size_t[3];
    index[1] = size_t(row);
    index[2] = size_t(col);

    for (int t = t1; t <= t2; t++)
    {
        index[0] = unsigned(t);
        if (var.type == NC_DOUBLE)
        {
            double value;
            nc_get_var1_double(ncId, idVar, index, &value);
            *buffer << getDateTimeStr(t) << ", " << value << endl;
        }
        if (var.type == NC_FLOAT)
        {
            float value;
            nc_get_var1_float(ncId, idVar, index, &value);
            *buffer << getDateTimeStr(t) << ", " << value << endl;
        }
        if (var.type <= NC_INT)
        {
            int value;
            nc_get_var1_int(ncId, idVar, index, &value);
            *buffer << getDateTimeStr(t) << ", " << double(value)/100 << endl;
        }
    }

    return true;
}


bool NetCDFHandler::createNewFile(std::string fileName)
{
    clear();

    int status = nc_create(fileName.data(), NC_CLOBBER|NC_NETCDF4, &ncId);
    return (status == NC_NOERR);
}


bool NetCDFHandler::writeMetadata(const gis::Crit3DLatLonHeader& latLonHeader, const string& title,
                                  const string& variableName, const string& variableUnit,
                                  const Crit3DDate& myDate, int nDays, int refYearStart, int refYearEnd)
{
    if (ncId == NODATA) return false;

    bool timeDimensionExists = (myDate != NO_DATE);
    bool boundsExist = false;
    bool referenceIntervalExists = false;
    if (nDays != 0 && nDays != NODATA)
    {
        boundsExist = true;
    }
    if (refYearStart != 0 && refYearStart != NODATA
        && refYearEnd != 0 && refYearEnd != NODATA)
    {
        referenceIntervalExists = true;
    }
    nrLat = latLonHeader.nrRows;
    nrLon = latLonHeader.nrCols;
    int varLat, varLon, status;
    int varTime = 0;
    int varTimeBounds = 0;

    // global attributes
    status = nc_put_att_text(ncId, NC_GLOBAL, "title", title.length(), title.c_str());
    if (status != NC_NOERR) return false;
    status = nc_put_att_text(ncId, NC_GLOBAL, "history", 11, "Version 1.0");
    if (status != NC_NOERR) return false;
    status = nc_put_att_text(ncId, NC_GLOBAL, "Conventions", 6, "CF-1.7");
    if (status != NC_NOERR) return false;

    // time
    if (timeDimensionExists)
    {
        status = nc_def_dim(ncId, "time", unsigned(1), &idTime);
        if (status != NC_NOERR) return false;

        status = nc_def_var (ncId, "time", NC_FLOAT, 1, &idTime, &varTime);
        if (status != NC_NOERR) return false;

        status = nc_put_att_text(ncId, varTime, "standard_name", 4, "time");
        if (status != NC_NOERR) return false;

        std::string timeUnits = "days since " + myDate.toStdString();
        status = nc_put_att_text(ncId, varTime, "units", timeUnits.length(), timeUnits.c_str());
        if (status != NC_NOERR) return false;

        std::string timeCalendarAtt = "gregorian" ;
        status = nc_put_att_text(ncId, varTime, "calendar", timeCalendarAtt.length(), timeCalendarAtt.c_str());
        if (status != NC_NOERR) return false;

        if (boundsExist)
        {
            status = nc_def_dim(ncId, "bnds", unsigned(2), &idTimeBnds);
            if (status != NC_NOERR) return false;

            int time_bnds_dims[2];
            time_bnds_dims[0] = idTime;
            time_bnds_dims[1] = idTimeBnds;
            status = nc_def_var (ncId, "time_bnds", NC_FLOAT, 2, time_bnds_dims, &varTimeBounds);
            if (status != NC_NOERR) return false;

            status = nc_put_att_text(ncId, varTime, "bounds", 9, "time_bnds");
            if (status != NC_NOERR) return false;
        }
    }

    // lat
    status = nc_def_dim(ncId, "lat", unsigned(nrLat), &idLat);
    if (status != NC_NOERR) return false;

    status = nc_def_var (ncId, "lat", NC_FLOAT, 1, &idLat, &varLat);
    if (status != NC_NOERR) return false;

    status = nc_put_att_text(ncId, varLat, "standard_name", 8, "latitude");
    if (status != NC_NOERR) return false;
    status = nc_put_att_text(ncId, varLat, "units", 13, "degrees_north");
    if (status != NC_NOERR) return false;

    // lon
    status = nc_def_dim(ncId, "lon", unsigned(nrLon), &idLon);
    if (status != NC_NOERR) return false;

    status = nc_def_var (ncId, "lon", NC_FLOAT, 1, &idLon, &varLon);
    if (status != NC_NOERR) return false;

    status = nc_put_att_text(ncId, varLon, "standard_name", 9, "longitude");
    if (status != NC_NOERR) return false;
    status = nc_put_att_text(ncId, varLon, "units", 12, "degrees_east");
    if (status != NC_NOERR) return false;

    // generic variable
    variables.resize(1);
    if (timeDimensionExists)
    {
        int nrDims = 3;
        int varDimId[3];
        varDimId[0] = idTime;
        varDimId[1] = idLat;
        varDimId[2] = idLon;

        status = nc_def_var (ncId, variableName.c_str(), NC_FLOAT, nrDims, varDimId, &(variables[0].id));
        if (status != NC_NOERR) return false;
    }
    else
    {
        int nrDims = 2;
        int varDimId[2];
        varDimId[0] = idLat;
        varDimId[1] = idLon;

        status = nc_def_var (ncId, variableName.c_str(), NC_FLOAT, nrDims, varDimId, &(variables[0].id));
        if (status != NC_NOERR) return false;
    }

    if (referenceIntervalExists)
    {
        std::string referenceYearStart = std::to_string(refYearStart);
        status = nc_put_att_text(ncId, variables[0].id, "reference_start_year", referenceYearStart.length(), referenceYearStart.c_str());
        if (status != NC_NOERR) return false;
        std::string referenceYearEnd = std::to_string(refYearEnd);
        status = nc_put_att_text(ncId, variables[0].id, "reference_end_year", referenceYearEnd.length(), referenceYearEnd.c_str());
        if (status != NC_NOERR) return false;
    }

    // attributes
    status = nc_put_att_text(ncId, variables[0].id, "long_name", variableName.length(), variableName.c_str());
    if (status != NC_NOERR) return false;

    // Units are not required for dimensionless quantities
    if (variableUnit != "")
    {
        status = nc_put_att_text(ncId, variables[0].id, "units", variableUnit.length(), variableUnit.c_str());
        if (status != NC_NOERR) return false;
    }

    // no data
    float missing[] = {NODATA};
    status = nc_put_att_float(ncId, variables[0].id, "missing_value", NC_FLOAT, 1, missing);
    if (status != NC_NOERR) return false;

    // compression
    int shuffle = NC_SHUFFLE;
    int deflate = 1;
    int deflate_level = 1;
    status = nc_def_var_deflate(ncId, variables[0].id, shuffle, deflate, deflate_level);
    if (status != NC_NOERR) return false;

    // valid range
    /*
    float range[] = {-1000.0, 1000.0};
    status = nc_put_att_float(ncId, variables[0].id, "valid_range", NC_FLOAT, 2, range);
    if (status != NC_NOERR) return false;
    */

    // end of metadata
    status = nc_enddef(ncId);
    if (status != NC_NOERR) return false;

    // write time
    if (timeDimensionExists)
    {

        float timeValue[1];
        timeValue[0] = float(nDays+1) / 2.f;
        status = nc_put_var_float(ncId, varTime, &timeValue[0]);
        if (status != NC_NOERR) return false;
        // time bounds
        if (boundsExist)
        {
            float boundsValue[2];
            boundsValue[0] = 0.0;
            boundsValue[1] = float(nDays);
            status = nc_put_var_float(ncId, varTimeBounds, &boundsValue[0]);
            if (status != NC_NOERR) return false;
        }
    }

    // set lat/lon arrays
    lat = new float[unsigned(nrLat)];
    lon = new float[unsigned(nrLon)];
    for (int row = 0; row < nrLat; row++)
    {
        lat[row] = float(latLonHeader.llCorner.latitude + latLonHeader.dy * (latLonHeader.nrRows - row - 0.5));
    }
    for (int col = 0; col < nrLon; col++)
    {
        lon[col] = float(latLonHeader.llCorner.longitude + latLonHeader.dx * (col + 0.5));
    }

    // write lat/lon vectors
    status = nc_put_var_float(ncId, varLat, lat);
    if (status != NC_NOERR) return false;

    status = nc_put_var_float(ncId, varLon, lon);
    if (status != NC_NOERR) return false;

    return true;
}


bool NetCDFHandler::writeData_NoTime(const gis::Crit3DRasterGrid& myDataGrid)
{
    if (ncId == NODATA) return false;

    float* var = new float[unsigned(nrLat*nrLon)];

    for (int row = 0; row < nrLat; row++)
    {
        for (int col = 0; col < nrLon; col++)
        {
            float value = myDataGrid.value[row][col];
            // check on not active cells (for meteo grid)
            if (isEqual(value, myDataGrid.header->flag) || isEqual(value, NO_ACTIVE))
                value = NODATA;

            var[row*nrLon + col] = value;
        }
    }

    int status = nc_put_var_float(ncId, variables[0].id, var);
    if (status != NC_NOERR) return false;

    return true;
}


bool NetCDFHandler::extractVariableMap(int idVar, const Crit3DTime& myTime, std::string& errorStr)
{
    // initialize
    for (int row = 0; row < dataGrid.header->nrRows; row++)
    {
        for (int col = 0; col < dataGrid.header->nrCols; col++)
        {
            dataGrid.value[row][col] = NODATA;
        }
    }

    // check variable
    NetCDFVariable currentVar = getVariableFromId(idVar);
    if (currentVar.getVarName() == "")
    {
        errorStr = "Wrong variable.";
        return false;
    }

    // check time
    if (! isTimeReadable())
    {
        errorStr = "Missing Time dimension.";
        return false;
    }

    if (myTime < getFirstTime() || myTime > getLastTime())
    {
        errorStr = "Time is out of range.";
        return false;
    }

    // search time index
    long timeIndex = NODATA;
    for (long i = 0; i < nrTime; i++)
    {
        if (getTime(i) == myTime)
        {
            timeIndex = i;
            break;
        }
    }
    if  (timeIndex == NODATA)
    {
        errorStr = "No available time index.";
        return false;
    }

    // read data
    int retVal;
    long nrValues;
    size_t start[] = {unsigned(timeIndex), 0, 0};
    size_t count[] = {1, 1, 1};
    if (isLatLon)
    {
        count[1] = nrLat;
        count[2] = nrLon;
        nrValues = nrLat * nrLon;
    }
    else
    {
        count[1] = nrX;
        count[2] = nrY;
        nrValues = nrX * nrY;
    }
    float* values = new float[nrValues];

    switch(currentVar.type)
    {
        case NC_DOUBLE:
        {
            double* valuesDouble = new double[nrValues];
            retVal = nc_get_vara_double(ncId, idVar, start, count, valuesDouble);
            for (int i=0; i < nrValues; i++)
            {
                values[i] = float(valuesDouble[i]);
            }
            delete[] valuesDouble;
            break;
        }
        case NC_FLOAT:
        {
            retVal = nc_get_vara_float(ncId, idVar, start, count, values);
            break;
        }
        case NC_INT:
        {
            int* valuesInt = new int[nrValues];
            retVal = nc_get_vara_int(ncId, idVar, start, count, valuesInt);
            for (int i=0; i < nrValues; i++)
            {
                values[i] = float(valuesInt[i]);
            }
            delete[] valuesInt;
            break;
        }
            default:
        {
            delete[] values;
            errorStr = "Wrong variable type.";
            return false;
        }
    }

    if (retVal != NC_NOERR)
    {
        errorStr.append(nc_strerror(retVal));
        delete[] values;
        return false;
    }
    else
    {
        for (int row = 0; row < dataGrid.header->nrRows; row++)
        {
            for (int col = 0; col < dataGrid.header->nrCols; col++)
            {
                if (isYincreasing)
                {
                    dataGrid.value[row][col] = values[(dataGrid.header->nrRows-row-1) * nrLon + col];
                }
                else
                {
                    dataGrid.value[row][col] = values[row * nrLon + col];
                }
            }
        }
        delete[] values;
        return true;
    }
}


gis::Crit3DRasterGrid* NetCDFHandler::getRaster()
{
    return &dataGrid;
}
