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
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include <algorithm>
#include <sstream>
#include <fstream>
#include <cstring>
#include <math.h>

#include "gis.h"

using namespace std;


bool splitKeyValue(const string &str, string &key, string &value)
{
    key = "";
    value = "";

    istringstream myStream(str);
    myStream >> key;
    myStream >> value;

    if (key == "" || value == "")
        return false;

    return true;
}


void cleanSpaces(std::string &str)
{
    str.erase(remove_if(str.begin(), str.end(), [](unsigned char c){ return isspace(c); }), str.end());
}


void cleanBraces(std::string &str)
{
    str.erase(remove_if(str.begin(), str.end(), [](unsigned char c){ return (c == '{' || c == '}'); }), str.end());
}


vector<string> splitCommaDelimited(const string &str)
{
    stringstream ss(str);
    vector<string> result;

    while(ss.good())
    {
        string substr;
        getline(ss, substr, ',');
        result.push_back(substr);
    }

    return result;
}


bool splitKeyValueByDelimiter(const string &myLine, const string &delimiter, string &key, string &value)
{
    key = "";
    value = "";

    key = myLine.substr(0, myLine.find(delimiter));
    value = myLine.substr(myLine.find(delimiter)+1);

    if (key == "" || value == "")
        return false;

    return true;
}


string upperCase(const string &myStr)
{
    string upperCaseStr = myStr;
    transform(myStr.begin(), myStr.end(), upperCaseStr.begin(), ::toupper);
    return upperCaseStr;
}


namespace gis
    {

    /*!
     * \brief Read a ESRI float header file (.hdr)
     * \param fileName    string
     * \param header      Crit3DRasterHeader pointer
     * \param error       string
     * \return true on success, false otherwise
     */
    bool readEsriGridHeader(const std::string &fileName, gis::Crit3DRasterHeader *header, std::string &errorStr)
    {
        string myLine, myKey, upKey, valueStr;
        int nrKeys = 0;

        // check suffix
        string fn = fileName;
        std::string key (".flt");
        std::size_t found = fn.rfind(key);
        if (found != std::string::npos)
            fn.replace (found, key.length(), "");

        // open file
        fn += ".hdr";
        ifstream  myFile(fn.c_str());

        if (myFile.fail())
        {
            errorStr = "Missing file: " + fn;
            return false;
        }

        while (myFile.good())
        {
            getline (myFile, myLine);
            if (splitKeyValue(myLine, myKey, valueStr))
            {
                upKey = upperCase(myKey);

                if ((upKey == "NCOLS") || (upKey == "NROWS") || (upKey == "CELLSIZE")
                    ||    (upKey == "XLLCORNER") || (upKey == "YLLCORNER")
                    ||    (upKey == "NODATA_VALUE") || (upKey == "NODATA"))
                    nrKeys++;

                if (upKey == "NCOLS")
                    header->nrCols = stoi(valueStr);

                else if (upKey == "NROWS")
                    header->nrRows = stoi(valueStr);

                else if (upKey == "DATATYPE")
                {
                    // non standard key - derived from envi
                    header->nrBytes = stoi(valueStr);
                    if (header->nrBytes > 4)
                    {
                        errorStr = "Wrong data type:" + valueStr + " The maximum allowed is 4 (float).";
                        return false;
                    }
                }

                // LLCORNER = Lower Left corner
                else if (upKey == "XLLCORNER")
                    header->llCorner.x = stod(valueStr);

                else if (upKey == "YLLCORNER")
                    header->llCorner.y = stod(valueStr);

                else if (upKey == "CELLSIZE")
                    header->cellSize = stod(valueStr);

                else if ((upKey == "NODATA_VALUE") || (upKey == "NODATA"))
                    header->flag = stof(valueStr);
            }
        }
        myFile.close();

        if (nrKeys < 6)
        {
            errorStr = "Missing keys in header file.";
            return false;
        }

        return true;
    }


    /*!
     * \brief Read a ENVI header file (.hdr)
     * \param fileName    string
     * \param header      Crit3DRasterHeader pointer
     * \param error       string
     * \return true on success, false otherwise
     */
    bool readEnviHeader(string fileName, gis::Crit3DRasterHeader *header, int currentUtmZone, string &errorStr)
    {
        errorStr = "";

        string completeFileName = fileName + ".hdr";
        ifstream  myFile(completeFileName.c_str());

        if (! myFile.is_open())
        {
            completeFileName = fileName + ".img.hdr";
            myFile.open(completeFileName.c_str());
        }

        if (! myFile.is_open())
        {
            errorStr = "Missing file: " + fileName + ".hdr";
            return false;
        }

        int nrKeys = 0;
        bool hasNoData = false;
        string myLine, key, valueStr;
        while (myFile.good())
        {
            getline (myFile, myLine);
            if (splitKeyValueByDelimiter(myLine, "=", key, valueStr))
            {
                // no spaces and uppercase for comparison
                cleanSpaces(key);
                string upKey = upperCase(key);

                if ((upKey == "SAMPLES") || (upKey == "LINES") || (upKey == "MAPINFO")
                    || ((upKey == "DATAIGNOREVALUE") || (upKey == "NODATA")) )
                    nrKeys++;

                if (upKey == "SAMPLES")
                    header->nrCols = stoi(valueStr);

                else if (upKey == "LINES")
                    header->nrRows = stoi(valueStr);

                else if (upKey == "DATATYPE")
                {
                    header->nrBytes = stoi(valueStr);
                    if (header->nrBytes > 4)
                    {
                        errorStr = "Wrong data type:" + valueStr + " The maximum allowed is 4 (float).";
                        return false;
                    }
                }

                else if (upKey == "MAPINFO")
                {
                    // remove the curly braces, split the values ​​and remove the spaces
                    cleanBraces(valueStr);
                    vector<string> infoStr = splitCommaDelimited(valueStr);
                    for (int i = 0; i < int(infoStr.size()); i++)
                    {
                        cleanSpaces(infoStr[i]);
                    }

                    // check key values nr
                    if (infoStr.size() < 10)
                    {
                        int nrMissing = 10 - int(infoStr.size());
                        string nrStr = to_string(nrMissing);
                        errorStr = "Missing " + nrStr + " key values in the map info (hdr file).";
                        return false;
                    }

                    // check projection and datum
                    if (upperCase(infoStr[0]) != "UTM")
                    {
                        errorStr = "Only UTM projection is allowed.";
                        return false;
                    }
                    string datum = upperCase(infoStr[9]);
                    if (datum != "WGS84" && datum != "WGS-84")
                    {
                        errorStr = "Only WGS-84 datum is allowed.";
                        return false;
                    }

                    // chek UTM zone
                    int utmZone = stoi(infoStr[7]);
                    if (utmZone != currentUtmZone)
                    {
                        errorStr = "UTM zone: " + infoStr[7] +"\nCurrent UTM zone is " + to_string(currentUtmZone);
                        return false;
                    }

                    // check cellsize
                    if (infoStr[5] != infoStr[6])
                    {
                        errorStr = "Different cell sizes on x and y are not allowed.";
                        return false;
                    }

                    header->cellSize = stod(infoStr[5]);
                    header->llCorner.x = stod(infoStr[3]);
                    double yTopLeftcorner = stod(infoStr[4]);
                    header->llCorner.y = yTopLeftcorner - (header->nrRows * header->cellSize);
                }

                else if ((upKey == "DATAIGNOREVALUE") || (upKey == "NODATA"))
                {
                    header->flag = float(::atof(valueStr.c_str()));
                    hasNoData = true;
                }
            }
        }
        myFile.close();

        if (nrKeys < 4)
        {
            if (! hasNoData)
            {
                errorStr += "Wrong header file: missing data ignore value.";
            }
            else
            {
                errorStr = "Wrong header file: missing samples, lines or map info.";
            }

            return false;
        }

        return true;
    }


    /*!
     * \brief Read a ESRI/ENVI float data file (.flt or .img)
     * \param fileName      string name file
     * \param rasterGrid    Crit3DRasterGrid pointer
     * \param error         string
     * \return true on success, false otherwise
     */
    bool readRasterFloatData(const string &fileName, gis::Crit3DRasterGrid *rasterGrid, string &error)
    {
        FILE* filePointer;

        if (! rasterGrid->initializeGrid())
        {
            error = "Memory error: file too big.";
            return false;
        }

        filePointer = fopen (fileName.c_str(), "rb" );
        if (filePointer == nullptr)
        {
            error = "Error in opening raster file.";
            return false;
        }

        if (rasterGrid->header->nrBytes == 4)
        {
            // float
            for (int row = 0; row < rasterGrid->header->nrRows; row++)
            {
                fread (rasterGrid->value[row], sizeof(float), unsigned(rasterGrid->header->nrCols), filePointer);
            }
        }
        else if (rasterGrid->header->nrBytes == 1)
        {
            // byte
            unsigned char *rowValues = new unsigned char[unsigned(rasterGrid->header->nrCols)];
            for (int row = 0; row < rasterGrid->header->nrRows; row++)
            {
                fread (rowValues, sizeof(unsigned char), unsigned(rasterGrid->header->nrCols), filePointer);
                for(int col = 0; col < rasterGrid->header->nrCols; col++)
                {
                    rasterGrid->value[row][col] = float(rowValues[col]);
                }
            }
            delete[] rowValues;
        }
        else if (rasterGrid->header->nrBytes == 2)
        {
            // short
            short int *rowValues = new short int[unsigned(rasterGrid->header->nrCols)];
            for (int row = 0; row < rasterGrid->header->nrRows; row++)
            {
                fread (rowValues, sizeof(short int), unsigned(rasterGrid->header->nrCols), filePointer);
                for(int col = 0; col < rasterGrid->header->nrCols; col++)
                {
                    rasterGrid->value[row][col] = float(rowValues[col]);
                }
            }
            delete[] rowValues;
        }

        fclose (filePointer);

        return true;
    }


    /*!
     * \brief Write a ESRI grid header file (.hdr)
     * \param fileName      file name
     * \param header        Crit3DRasterHeader pointer
     * \param errorStr      error string
     * \return true on success, false otherwise
     */
    bool writeEsriGridHeader(const std::string &fileName, gis::Crit3DRasterHeader *header, std::string &errorStr)
    {
        std::string myFileName = fileName + ".hdr";
        std::ofstream myFile (myFileName.c_str());

        if (myFile.fail())
        {
            errorStr = "Error writing file: " + myFileName + '\n' + strerror(errno);
            return false;
        }

        myFile << "ncols         " << header->nrCols << "\n";
        myFile << "nrows         " << header->nrRows << "\n";

        char* xllcorner = new char[20];
        char* yllcorner = new char[20];
        sprintf(xllcorner, "%.03f", header->llCorner.x);
        sprintf(yllcorner, "%.03f", header->llCorner.y);

        myFile << "xllcorner     " << xllcorner << "\n";

        myFile << "yllcorner     " << yllcorner << "\n";

        myFile << "cellsize      " << header->cellSize << "\n";

        // different version of NODATA
        myFile << "NODATA_value  " << header->flag << "\n";
        myFile << "NODATA        " << header->flag << "\n";

        // crucial information
        myFile << "byteorder     LSBFIRST" << "\n";

        myFile.close();
        delete [] xllcorner;
        delete [] yllcorner;

        return true;
    }


    /*!
     * \brief Write a ESRI grid data file (.flt)
     * \param fileName    file name
     * \param myGrid      Crit3DRasterGrid pointer
     * \param errorStr    error string
     * \return true on success, false otherwise
     */
    bool writeEsriGridFlt(const std::string &fileName, Crit3DRasterGrid *myGrid, std::string &errorStr)
    {
        std::string myFileName = fileName + ".flt";
        std::FILE* filePointer;

        filePointer = fopen(myFileName.c_str(), "wb" );
        if (filePointer == nullptr)
        {
            errorStr = "Error writing file: " + myFileName + '\n' + strerror(errno);
            return false;
        }

        for (int row = 0; row < myGrid->header->nrRows; row++)
        {
            fwrite(myGrid->value[row], sizeof(float), unsigned(myGrid->header->nrCols), filePointer);
        }

        fclose (filePointer);
        return true;
    }


    /*!
     * \brief Write a ESRI float raster (.hdr and .flt)
     * \return true on success, false otherwise
     */
    bool writeEsriGrid(const std::string &fileName, Crit3DRasterGrid* rasterGrid, std::string &errorStr)
    {
        if (! gis::writeEsriGridHeader(fileName, rasterGrid->header, errorStr))
            return false;

        if (! gis::writeEsriGridFlt(fileName, rasterGrid, errorStr))
            return false;

        return true;
    }


    /*!
     * \brief Read a ESRI float raster (.hdr and .flt)
     * \return true on success, false otherwise
     */
    bool readEsriGrid(const string &fileName, Crit3DRasterGrid* rasterGrid, string &errorStr)
    {
        if (rasterGrid == nullptr)
            return false;
        rasterGrid->clear();

        if( gis::readEsriGridHeader(fileName, rasterGrid->header, errorStr) )
        {
            // check suffix
            std::size_t found = fileName.rfind(".flt");
            string fltFileName;
            if (found != std::string::npos)
            {
                fltFileName = fileName;
            }
            else
            {
                fltFileName = fileName + ".flt";
            }

            if (gis::readRasterFloatData(fltFileName, rasterGrid, errorStr))
            {
                gis::updateMinMaxRasterGrid(rasterGrid);
                rasterGrid->isLoaded = true;
            }
        }

        return rasterGrid->isLoaded;
    }


    /*!
     * \brief Read a ENVI grid data file (.hdr and .img)
     * \return true on success, false otherwise
     */
    bool readEnviGrid(string fileName, Crit3DRasterGrid* rasterGrid, int currentUtmZone, string &error)
    {
        rasterGrid->clear();

        if(gis::readEnviHeader(fileName, rasterGrid->header, currentUtmZone, error))
        {
            fileName += ".img";
            if (gis::readRasterFloatData(fileName, rasterGrid, error))
            {
                gis::updateMinMaxRasterGrid(rasterGrid);
                rasterGrid->isLoaded = true;
            }
        }

        return rasterGrid->isLoaded;
    }


    /*!
     * \brief Read a ESRI/ENVI float raster (header and data)
     * \return true on success, false otherwise
     */
    bool openRaster(string fileName, Crit3DRasterGrid *rasterGrid, int currentUtmZone, string &errorStr)
    {
        if (fileName.size() <= 4)
        {
            errorStr = "Wrong filename.";
            return false;
        }

        std::string fileNameWithoutExt = fileName.substr(0, fileName.size() - 4);
        std::string fileExtension = fileName.substr(fileName.size() - 4);

        bool isOk = false;
        if (fileExtension == ".flt")
        {
            isOk = gis::readEsriGrid(fileNameWithoutExt, rasterGrid, errorStr);
        }
        else if (fileExtension == ".img")
        {
            isOk = gis::readEnviGrid(fileNameWithoutExt, rasterGrid, currentUtmZone, errorStr);
        }
        else
        {
            errorStr = "Wrong suffix: .flt or .img are allowed.";
            isOk = false;
        }

        return isOk;
    }


    /*!
     * \brief Write a ENVI grid data file (.img and .hdr)
     * \return true on success, false otherwise
     */
    bool writeEnviGrid(string fileName, int utmZone, Crit3DRasterGrid *rasterGrid, string &error)
    {
        FILE* filePointer;
        string imgFileName = fileName + ".img";

        filePointer = fopen(imgFileName.c_str(), "wb" );
        if (filePointer == nullptr)
        {
            error = "error in writing file: " + imgFileName;
            return false;
        }

        // write grid
        for (int row = 0; row < rasterGrid->header->nrRows; row++)
            fwrite(rasterGrid->value[row], sizeof(float), unsigned(rasterGrid->header->nrCols), filePointer);

        fclose (filePointer);

        // write header file
        string headerFileName = fileName + ".hdr";
        ofstream myFile (headerFileName.c_str());

        if (!myFile.is_open())
        {
            error = "error in writing file: " + headerFileName;
            return false;
        }

        myFile << "ENVI\n";
        myFile << "description = {raster grid}\n";
        myFile << "samples = " << rasterGrid->header->nrCols << "\n";
        myFile << "lines = " << rasterGrid->header->nrRows << "\n";
        myFile << "bands = 1\n";
        myFile << "header offset = 0\n";
        myFile << "file type = ENVI Standard\n";
        myFile << "data type = 4\n";
        myFile << "interleave = bsq\n";
        myFile << "byte order = 0\n";
        myFile << "data ignore value = " << rasterGrid->header->flag << "\n";

        // top-left corner
        double yTopLeftcorner = rasterGrid->header->llCorner.y + rasterGrid->header->nrRows * rasterGrid->header->cellSize;
        myFile << "map info = {UTM, 1, 1, " << to_string(rasterGrid->header->llCorner.x) << ", " << to_string(yTopLeftcorner) << ", ";
        myFile << rasterGrid->header->cellSize << ", " << rasterGrid->header->cellSize << ", ";
        myFile << utmZone << ", North, WGS-84, units=Meters}";

        myFile.close();
        return true;
    }


    bool getGeoExtentsFromUTMHeader(const Crit3DGisSettings& mySettings, Crit3DRasterHeader *utmHeader, Crit3DLatLonHeader *latLonHeader)
    {
        Crit3DGeoPoint v[4];

        // compute vertexes
        Crit3DUtmPoint myVertex = utmHeader->llCorner;
        gis::getLatLonFromUtm(mySettings, myVertex, v[0]);
        myVertex.x += utmHeader->nrCols * utmHeader->cellSize;
        gis::getLatLonFromUtm(mySettings, myVertex, v[1]);
        myVertex.y += utmHeader->nrRows * utmHeader->cellSize;
        gis::getLatLonFromUtm(mySettings, myVertex, v[2]);
        myVertex.x = utmHeader->llCorner.x;
        gis::getLatLonFromUtm(mySettings, myVertex, v[3]);

        // compute LLcorner and URcorner
        Crit3DGeoPoint LLcorner, URcorner;
        LLcorner.longitude = min(v[0].longitude, v[3].longitude);
        URcorner.longitude = max(v[1].longitude, v[2].longitude);
        if (mySettings.startLocation.latitude >= 0)
        {
            LLcorner.latitude = min(v[0].latitude, v[1].latitude);
            URcorner.latitude = max(v[2].latitude, v[3].latitude);
        }
        else
        {
            LLcorner.latitude = max(v[0].latitude, v[1].latitude);
            URcorner.latitude = min(v[2].latitude, v[3].latitude);
        }

        // rowcol nr
        // increase resolution
        latLonHeader->nrRows = int(utmHeader->nrRows * 1.0);
        latLonHeader->nrCols = int(utmHeader->nrCols * 1.0);

        // dx, dy
        latLonHeader->dx = (URcorner.longitude - LLcorner.longitude) / latLonHeader->nrCols;
        latLonHeader->dy = (URcorner.latitude - LLcorner.latitude) / latLonHeader->nrRows;

        latLonHeader->llCorner.latitude = LLcorner.latitude;
        latLonHeader->llCorner.longitude = LLcorner.longitude;

        // flag
        latLonHeader->flag = utmHeader->flag;

        return true;
    }

    bool getGeoExtentsFromLatLonHeader(const Crit3DGisSettings& mySettings, double cellSize, Crit3DRasterHeader *utmHeader, Crit3DLatLonHeader *latLonHeader)
    {
        Crit3DUtmPoint v[4];

        // compute vertexes
        gis::Crit3DGeoPoint geoPoint;

        // LL
        geoPoint.latitude = latLonHeader->llCorner.latitude;
        geoPoint.longitude = latLonHeader->llCorner.longitude;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[0]);

        // LR
        geoPoint.longitude = latLonHeader->llCorner.longitude + latLonHeader->nrCols * latLonHeader->dx;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[1]);

        // UR
        geoPoint.latitude = latLonHeader->llCorner.latitude + latLonHeader->nrRows * latLonHeader->dy;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[2]);

        // UL
        geoPoint.longitude = latLonHeader->llCorner.longitude;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[3]);

        double xmin = floor(min(v[0].x, v[3].x));
        double xmax = floor(max(v[1].x, v[2].x)) +1.;
        double ymin = floor(min(v[0].y, v[1].y));
        double ymax = floor(max(v[2].y, v[3].y)) +1.;

        utmHeader->cellSize = cellSize;
        utmHeader->nrCols = int(floor((xmax-xmin)/utmHeader->cellSize) + 1);
        utmHeader->nrRows = int(floor((ymax-ymin)/utmHeader->cellSize) + 1);
        utmHeader->llCorner.x = xmin;
        utmHeader->llCorner.y = ymin;

        utmHeader->flag = latLonHeader->flag;
        return true;
    }

    double getGeoCellSizeFromLatLonHeader(const Crit3DGisSettings& mySettings, Crit3DLatLonHeader *latLonHeader)
    {
        Crit3DUtmPoint v[4];

        // compute vertexes
        gis::Crit3DGeoPoint geoPoint;

        // LL
        geoPoint.latitude = latLonHeader->llCorner.latitude;
        geoPoint.longitude = latLonHeader->llCorner.longitude;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[0]);

        // LR
        geoPoint.longitude = latLonHeader->llCorner.longitude + latLonHeader->nrCols * latLonHeader->dx;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[1]);

        // UR
        geoPoint.latitude = latLonHeader->llCorner.latitude + latLonHeader->nrRows * latLonHeader->dy;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[2]);

        // UL
        geoPoint.longitude = latLonHeader->llCorner.longitude;
        gis::getUtmFromLatLon(mySettings.utmZone, geoPoint, &v[3]);

        double xCellSize = (v[1].x-v[0].x)/latLonHeader->nrCols;
        double yCellSize = (v[3].y-v[0].y)/latLonHeader->nrRows;

        return min(xCellSize,yCellSize);
    }

}
