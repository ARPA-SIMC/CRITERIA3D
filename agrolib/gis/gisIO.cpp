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
#include <math.h>
#include "commonConstants.h"
#include "gis.h"

using namespace std;

bool splitKeyValue(std::string myLine, std::string *myKey, std::string *myValue)
{
    *myKey = ""; *myValue = "";
    istringstream myStream(myLine);

    myStream >> *myKey;
    myStream >> *myValue;
    if ((*myKey == "") || (*myValue == "")) return (false);
    else return(true);
}

std::string upperCase(std::string myStr)
{
    std::string upperCaseStr = myStr;
    transform(myStr.begin(), myStr.end(), upperCaseStr.begin(), ::toupper);
    return(upperCaseStr);
}

namespace gis
    {

    /*!
     * \brief Read a ESRI grid header file (.hdr)
     * \param myFileName    string
     * \param myHeader      Crit3DRasterHeader pointer
     * \param myError       string pointer
     * \return true on success, false otherwise
     */
    bool readEsriGridHeader(string myFileName, gis::Crit3DRasterHeader *myHeader, string* myError)
    {
        string myLine, myKey, upKey, myValue;
        int nrKeys = 0;

        myFileName += ".hdr";
        ifstream  myFile (myFileName.c_str());

        if (!myFile.is_open())
        {
            *myError = "Missing file: " + myFileName;
            return false;
        }

        while (myFile.good())
        {
            getline (myFile, myLine);
            if (splitKeyValue(myLine, &myKey, &myValue))
            {
                upKey = upperCase(myKey);

                if ((upKey == "NCOLS") || (upKey == "NROWS") || (upKey == "CELLSIZE")
                    ||    (upKey == "XLLCORNER") || (upKey == "YLLCORNER")
                    ||    (upKey == "NODATA_VALUE") || (upKey == "NODATA"))
                    nrKeys++;

                if (upKey == "NCOLS")
                    myHeader->nrCols = ::atoi(myValue.c_str());

                else if (upKey == "NROWS")
                    myHeader->nrRows = ::atoi(myValue.c_str());

                // LLCORNER is the lower-left corner
                else if (upKey == "XLLCORNER")
                    myHeader->llCorner.x = ::atof(myValue.c_str());

                else if (upKey == "YLLCORNER")
                    myHeader->llCorner.y = ::atof(myValue.c_str());

                else if (upKey == "CELLSIZE")
                    myHeader->cellSize = ::atof(myValue.c_str());

                else if ((upKey == "NODATA_VALUE") || (upKey == "NODATA"))
                    myHeader->flag = float(::atof(myValue.c_str()));
            }
        }
        myFile.close();

        if (nrKeys < 6)
        {
            *myError = "Missing keys in .hdr file.";
            return(false);
        }
        return(true);
    }


    /*!
     * \brief Read a ESRI grid data file (.flt)
     * \param fileName string name file
     * \param myGrid Crit3DRasterGrid pointer
     * \param myError string pointer
     * \return true on success, false otherwise
     */
    bool readEsriGridFlt(string fileName, gis::Crit3DRasterGrid *myGrid, string *myError)
    {
        fileName += ".flt";

        FILE* filePointer;

        if (! myGrid->initializeGrid())
        {
            *myError = "Memory error: file too big.";
            return(false);
        }

        filePointer = fopen (fileName.c_str(), "rb" );
        if (filePointer == nullptr)
        {
            *myError = "File .flt error.";
            return(false);
        }

        for (int row = 0; row < myGrid->header->nrRows; row++)
            fread (myGrid->value[row], sizeof(float), unsigned(myGrid->header->nrCols), filePointer);

        fclose (filePointer);

        return (true);
    }


    /*!
     * \brief Write a ESRI grid header file (.hdr)
     * \param myFileName string name file
     * \param myHeader Crit3DRasterHeader pointer
     * \param myError string pointer
     * \return true on success, false otherwise
     */
    bool writeEsriGridHeader(string myFileName, gis::Crit3DRasterHeader *myHeader, string* myError)
    {
        myFileName += ".hdr";
        ofstream myFile (myFileName.c_str());

        if (!myFile.is_open())
        {
            *myError = "File .hdr error.";
            return(false);
        }

        myFile << "ncols         " << myHeader->nrCols << "\n";
        myFile << "nrows         " << myHeader->nrRows << "\n";

        char* xllcorner = new char[20];
        char* yllcorner = new char[20];
        sprintf(xllcorner, "%.03f", myHeader->llCorner.x);
        sprintf(yllcorner, "%.03f", myHeader->llCorner.y);

        myFile << "xllcorner     " << xllcorner << "\n";

        myFile << "yllcorner     " << yllcorner << "\n";

        myFile << "cellsize      " << myHeader->cellSize << "\n";

        // different version of NODATA
        myFile << "NODATA_value  " << myHeader->flag << "\n";
        myFile << "NODATA        " << myHeader->flag << "\n";

        // crucial information
        myFile << "byteorder     LSBFIRST" << "\n";

        myFile.close();
        delete [] xllcorner;
        delete [] yllcorner;

        return true;
    }


    /*!
     * \brief Write a ESRI grid data file (.flt)
     * \param myFileName string name file
     * \param myHeader Crit3DRasterHeader pointer
     * \param myError string pointer
     * \return true on success, false otherwise
     */
    bool writeEsriGridFlt(string myFileName, gis::Crit3DRasterGrid *myGrid, string *myError)
    {
        myFileName += ".flt";

        FILE* filePointer;

        filePointer = fopen(myFileName.c_str(), "wb" );
        if (filePointer == nullptr)
        {
            *myError = "File .flt error.";
            return(false);
        }

        for (int myRow = 0; myRow < myGrid->header->nrRows; myRow++)
            fwrite(myGrid->value[myRow], sizeof(float), unsigned(myGrid->header->nrCols), filePointer);

        fclose (filePointer);
        return (true);
    }

    bool readEsriGrid(string myFileName, Crit3DRasterGrid* myGrid, string* myError)
    {
        if (myGrid == nullptr) return false;
        myGrid->isLoaded = false;

        Crit3DRasterHeader *myHeader;
        myHeader = new Crit3DRasterHeader;

        if(gis::readEsriGridHeader(myFileName, myHeader, myError))
        {
            myGrid->clear();
            *(myGrid->header) = *myHeader;

            if (gis::readEsriGridFlt(myFileName, myGrid, myError))
            {
                myGrid->isLoaded = true;
                updateMinMaxRasterGrid(myGrid);
            }
        }

        return(myGrid->isLoaded);
    }

    bool writeEsriGrid(string myFileName, Crit3DRasterGrid *myGrid, string *myError)
    {
        if (gis::writeEsriGridHeader(myFileName, myGrid->header, myError))
            if (gis::writeEsriGridFlt(myFileName, myGrid, myError))
                return(true);

        return(false);
    }


    bool getGeoExtentsFromUTMHeader(const Crit3DGisSettings& mySettings, Crit3DRasterHeader *utmHeader, Crit3DGridHeader *latLonHeader)
    {
        Crit3DGeoPoint v[4];

        // compute vertexes
        Crit3DUtmPoint myVertex = utmHeader->llCorner;
        gis::getLatLonFromUtm(mySettings, myVertex, &(v[0]));
        myVertex.x += utmHeader->nrCols * utmHeader->cellSize;
        gis::getLatLonFromUtm(mySettings, myVertex, &(v[1]));
        myVertex.y += utmHeader->nrRows * utmHeader->cellSize;
        gis::getLatLonFromUtm(mySettings, myVertex, &(v[2]));
        myVertex.x = utmHeader->llCorner.x;
        gis::getLatLonFromUtm(mySettings, myVertex, &(v[3]));

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

    bool getGeoExtentsFromLatLonHeader(const Crit3DGisSettings& mySettings, double cellSize, Crit3DRasterHeader *utmHeader, Crit3DGridHeader *latLonHeader)
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

    double getGeoCellSizeFromLatLonHeader(const Crit3DGisSettings& mySettings, Crit3DGridHeader *latLonHeader)
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
