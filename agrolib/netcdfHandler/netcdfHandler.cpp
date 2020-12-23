/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
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
    : name{""}, longName{""}, id{NODATA}, type{NODATA}
{ }


NetCDFVariable::NetCDFVariable(char* myName, int myId, int myType)
    : name{myName}, longName{myName}, id{myId}, type{myType}
{ }


std::string NetCDFVariable::getVarName()
{
    if (longName.size() <= 32)
        return longName;
    else
        return name;
}


NetCDFHandler::NetCDFHandler()
{
    ncId = NODATA;

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

    clear();
}


void NetCDFHandler::clear()
{
    if (x != nullptr) delete [] x;
    if (y != nullptr) delete [] y;
    if (lat != nullptr) delete [] lat;
    if (lon != nullptr) delete [] lon;
    if (time != nullptr) delete [] time;

    utmZone = NODATA;

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

    isLatLon = false;
    isLatDecreasing = false;
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
    metadata.clear();
}


void NetCDFHandler::initialize(int _utmZone)
{
    this->close();
    utmZone = _utmZone;
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


NetCDFVariable NetCDFHandler::getVariable(int idVar)
{
    for (unsigned int i = 0; i < variables.size(); i++)
        if (variables[i].id == idVar)
            return variables[i];

    return NetCDFVariable();
}


std::string NetCDFHandler::getVarName(int idVar)
{
    NetCDFVariable var = getVariable(idVar);
    return var.getVarName();
}


bool NetCDFHandler::isLoaded()
{
    return (variables.size() > 0);
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


std::string NetCDFHandler::getMetadata()
{
    return metadata.str();
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

    int nrDays, residualTime;

    if (isStandardTime)
    {
        long nrSeconds = long(time[timeIndex]);
        nrDays = int(floor(nrSeconds / DAY_SECONDS));
        residualTime = nrSeconds - (nrDays * DAY_SECONDS);
    }
    else if (isHourly)
    {
        long nrHours = long(time[timeIndex]);
        nrDays = int(floor(nrHours / 24));
        residualTime = (nrHours - nrDays*24) * HOUR_SECONDS;
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
    /*char* name = new char[NC_MAX_NAME+1];
    char* attrName = new char[NC_MAX_NAME+1];
    char* varName = new char[NC_MAX_NAME+1];
    char* typeName = new char[NC_MAX_NAME+1];*/

    char* valueStr;
    int valueInt;
    double value;
    size_t length;
    nc_type ncTypeId;
    int timeType = NC_DOUBLE;

    //NC_NOWRITE tells netCDF we want read-only access
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
    metadata << fileName << endl << endl;
    metadata << "Global attributes:" << endl;

    for (int a = 0; a < nrGlobalAttributes; a++)
    {
        nc_inq_attname(ncId, NC_GLOBAL, a, attrName);
        nc_inq_attlen(ncId, NC_GLOBAL, attrName, &length);

        valueStr = new char[length+1];
        nc_get_att_text(ncId, NC_GLOBAL, attrName, valueStr);

        string myString = string(valueStr).substr(0, length);
        metadata << attrName << " = " << myString << endl;

        delete [] valueStr;
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
           isLatLon = false;
       }
       else if (lowerCase(string(name)) == "y")
       {
           nrY = int(length);
           isLatLon = false;
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

       metadata << i << " - " << name << "\t values: " << length << endl;
   }

   if (isLatLon)
   {
       metadata <<"\n(lon,lat) = "<< nrLon << "," <<nrLat << endl;
   }
   else
   {
       metadata <<"\n(x,y) = "<< nrX << "," << nrY << endl;
   }

   // VARIABLES
   metadata << "\nVariables: " << endl;
   for (int v = 0; v < nrVariables; v++)
   {
       nc_inq_var(ncId, v, varName, &ncTypeId, &nrVarDimensions, varDimIds, &nrVarAttributes);
       nc_inq_type(ncId, ncTypeId, typeName, &length);

       // is Variable?
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
       else if (lowerCase(string(varName)) == "x")
           idX = v;
       else if (lowerCase(string(varName)) == "y")
           idY = v;
       else if (lowerCase(string(varName)) == "lat" || lowerCase(string(varName)) == "latitude")
           idLat = v;
       else if (lowerCase(string(varName)) == "lon" || lowerCase(string(varName)) == "longitude")
           idLon = v;

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
                valueStr = new char[length+1];
                nc_get_att_text(ncId, v, attrName, valueStr);

                string myString = string(valueStr).substr(0, length);
                metadata << attrName << " = " << myString << endl;

                if (v == idTime)
                {
                    if (lowerCase(string(attrName)) == "units")
                    {
                        if (lowerCase(myString).substr(0, 18) == "seconds since 1970")
                        {
                            isStandardTime = true;
                            firstDate = Crit3DDate(1, 1, 1970);
                        }
                        else if (lowerCase(myString).substr(0, 11) == "hours since")
                        {
                            isHourly = true;
                            std::string dateStr = lowerCase(myString).substr(12, 21);
                            firstDate = Crit3DDate(dateStr);
                        }
                        else if (lowerCase(myString).substr(0, 10) == "days since")
                        {
                            isDaily = true;
                            std::string dateStr = lowerCase(myString).substr(11, 20);
                            firstDate = Crit3DDate(dateStr);
                        }
                    }
                }
                if (lowerCase(string(attrName)) == "long_name")
                    setVarLongName(string(varName), myString);

                delete [] valueStr;
            }
            else if (ncTypeId == NC_INT)
            {
                nc_get_att(ncId, v, attrName, &valueInt);
                metadata << attrName << " = " << valueInt << endl;
            }
            else if (ncTypeId == NC_DOUBLE)
            {
                nc_get_att(ncId, v, attrName, &value);
                metadata << attrName << " = " << value << endl;
            }
        }
    }

    if (isLatLon)
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

            if (lat[1] > lat[0])
            {
                latLonHeader.llCorner.latitude = double(lat[0]);
                latLonHeader.dy = double(lat[nrLat-1]-lat[0]) / double(nrLat-1);
                isLatDecreasing = false;
            }
            else
            {
                latLonHeader.llCorner.latitude = double(lat[nrLat-1]);
                latLonHeader.dy = double(lat[0]-lat[nrLat-1]) / double(nrLat-1);
                isLatDecreasing = true;
            }

            latLonHeader.llCorner.longitude -= (latLonHeader.dx * 0.5);
            latLonHeader.llCorner.latitude -= (latLonHeader.dy * 0.5);

            latLonHeader.flag = NODATA;

            dataGrid.header->convertFromLatLon(latLonHeader);
            dataGrid.initializeGrid(0);
        }
    }
    else
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
            dataGrid.header->llCorner.x = double(x[0]) - dataGrid.header->cellSize*0.5;
            dataGrid.header->llCorner.y = double(y[0]) - dataGrid.header->cellSize*0.5;

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
        else if (timeType == NC_INT)
        {
            int* intTime = new int[unsigned(nrTime)];
            retval = nc_get_var_int(ncId, idTime, intTime);
            for (int i = 0; i < nrTime; i++)
            {
                time[i] = double(intTime[i]);
            }
            delete [] intTime;
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


bool NetCDFHandler::exportDataSeries(int idVar, gis::Crit3DGeoPoint geoPoint, Crit3DTime firstTime, Crit3DTime lastTime, stringstream *buffer)
{
    // check
    if (! isTimeReadable())
    {
        *buffer << "Wrong time dimension! Use standard POSIX (seconds since 1970-01-01)." << endl;
        return false;
    }
    if (! isPointInside(geoPoint))
    {
        *buffer << "Wrong position!" << endl;
        return false;
    }

    if (firstTime < getFirstTime() || lastTime > getLastTime())
    {
        *buffer << "Wrong time index!" << endl;
        return false;
    }

    // find row, col
    int row, col;
    if (isLatLon)
    {
        gis::getRowColFromLatLon(latLonHeader, geoPoint, &row, &col);
        if (!isLatDecreasing)
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
        if (getTime(i) == firstTime)
            t1 = i;
        if (getTime(i) == lastTime)
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
    NetCDFVariable var = getVariable(idVar);
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

    int status = nc_create(fileName.data(), NC_CLOBBER, &ncId);
    return (status == NC_NOERR);
}


bool NetCDFHandler::writeGeoDimensions(const gis::Crit3DGridHeader& latLonHeader)
{
    if (ncId == NODATA) return false;

    nrLat = latLonHeader.nrRows;
    nrLon = latLonHeader.nrCols;
    int varLat, varLon;

    // def dimensions (lat/lon)
    int status = nc_def_dim(ncId, "latitude", unsigned(nrLat), &idLat);
    if (status != NC_NOERR) return false;

    status = nc_def_dim(ncId, "longitude", unsigned(nrLon), &idLon);
    if (status != NC_NOERR) return false;

    // def geo variables (lat/lon)
    status = nc_def_var (ncId, "latitude", NC_FLOAT, 1, &idLat, &varLat);
    if (status != NC_NOERR) return false;

    status = nc_def_var (ncId, "longitude", NC_FLOAT, 1, &idLon, &varLon);
    if (status != NC_NOERR) return false;

    // def generic variable
    variables.resize(1);
    int varDimId[2];
    varDimId[0] = idLat;
    varDimId[1] = idLon;
    status = nc_def_var (ncId, "var", NC_FLOAT, 2, varDimId, &(variables[0].id));
    if (status != NC_NOERR) return false;

    // attributes
    status = nc_put_att_text(ncId, varLat, "units", 13, "degrees_north");
    if (status != NC_NOERR) return false;

    status = nc_put_att_text(ncId, varLon, "units", 12, "degrees_east");
    if (status != NC_NOERR) return false;

    // valid range
//    float range[] = {-1000.0, 1000.0};
//    status = nc_put_att_float(ncId, variables[0].id, "valid_range", NC_FLOAT, 2, range);
//    if (status != NC_NOERR) return false;

    // no data
    float missing[] = {float(NODATA)};
    status = nc_put_att_float(ncId, variables[0].id, "missing_value", NC_FLOAT, 1, missing);
    if (status != NC_NOERR) return false;

    // end of metadata
    status = nc_enddef(ncId);
    if (status != NC_NOERR) return false;

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
            var[row*nrLon + col] = myDataGrid.value[row][col];
        }
    }

    int status = nc_put_var_float(ncId, variables[0].id, var);
    if (status != NC_NOERR) return false;

    return true;
}


bool NetCDFHandler::extractVariableMap(int idVar, Crit3DTime myTime, gis::Crit3DRasterGrid* myDataGrid, string *error)
{
    // check variable
    NetCDFVariable var = getVariable(idVar);
    if (var.getVarName() == "")
    {
        *error = "Wrong variable!";
        return false;
    }

    // check time
    int timeIndex = NODATA;
    if (isTimeReadable())
    {
        if (myTime < getFirstTime() || myTime > getLastTime())
        {
            *error = "Time is out of range.";
            return false;
        }

        // search time index
        int i = 0;
        while (i < nrTime && timeIndex == NODATA)
        {
            if (getTime(i) == myTime)
                timeIndex = i;
            i++;
        }
        if  (timeIndex == NODATA)
        {
            *error = "Data not found for this time.";
            return false;
        }
    }

    // read data
    // todo:  nc_get_vara_float
    int retval = nc_get_var_float(ncId, idVar, &myDataGrid->value[0][0]);
    if (retval != NC_NOERR)
    {
        error->append(nc_strerror(retval));
        return false;
    }

    return true;
}
