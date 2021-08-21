/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,

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

/*! TestSolarRadiation
 * compute a map of global solar irradiance (clear sky)
 * for a specified date/time,
 * input: a Digital Elevation Model (ESRI .flt)
*/

#include <QCoreApplication>
#include <QDir>
#include "iostream"
#include "commonConstants.h"
#include "gis.h"
#include "radiationSettings.h"
#include "solarRadiation.h"


bool searchDefaultPath(QString inputPath, QString* outputPath)
{
    QString myPath = inputPath;
    QString myVolumeDOS = inputPath.left(3);

    bool isFound = false;
    while (! isFound)
    {
        if (QDir(myPath + "DATA").exists())
        {
            isFound = true;
            break;
        }

        if (QDir::cleanPath(myPath) == "/" || QDir::cleanPath(myPath) == myVolumeDOS)
            break;

        myPath += "../";
    }

    if (! isFound)
    {
        std::cout << "\nDATA directory is missing";
        return false;
    }

    *outputPath = QDir::cleanPath(myPath) + "/DATA/";
    return true;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::cout << "\nTEST Solar Radiation library" << std::endl;
    std::cout << "Compute a map of global solar irradiance (clear sky)" << std::endl;

    // GIS SETTINGS (UTM zone, time zone)
    gis::Crit3DGisSettings* gisSettings = new gis::Crit3DGisSettings();
    gisSettings->utmZone = 32;
    gisSettings->timeZone = 1;

    // DATETIME (UTC time)
    Crit3DDate* myDate = new Crit3DDate(1, 7, 2018);
    int myHour = 12;

    std::cout << "UTM zone: " << gisSettings->utmZone << std::endl;
    std::cout << "Date: " << myDate->toStdString() << " hour: " << myHour << " UTC" << std::endl;

    // Digital Elevation Model
    QString path;
    QString appPath = a.applicationDirPath() + "/";
    if (! searchDefaultPath(appPath, &path)) return -1;
    std::string inputFileName = path.toStdString() + + "DEM/dem_ravone";

    gis::Crit3DRasterGrid* DEM = new gis::Crit3DRasterGrid();
    std::string* error = new std::string();
    if (gis::readEsriGrid(inputFileName, DEM, error))
        std::cout << "\nDEM = " << inputFileName << std::endl;
    else
    {
        std::cout << "Error in reading:" << inputFileName << std::endl << *error << std::endl;
        return 0;
    }

    // SET RADIATION SETTINGS
    Crit3DRadiationSettings* radSettings = new Crit3DRadiationSettings();
    radSettings->setGisSettings(gisSettings);

    // INITIALIZE RADIATION MAPS (deafult trasmissivity = 0.75)
    Crit3DRadiationMaps* radMaps = new Crit3DRadiationMaps(*DEM, *gisSettings);

    int mySeconds = HOUR_SECONDS * myHour;
    Crit3DTime* myTime = new Crit3DTime(*myDate, mySeconds);

    std::cout << "\nComputing..." << std::endl;

    // COMPUTE POTENTIAL GLOBAL RADIATION MAPS
    if (radiation::computeRadiationGridPresentTime(radSettings, *DEM, radMaps, *myTime))
        std::cout << "\nGlobal solar irradiance (clear sky) computed." << std::endl;
    else
        std::cout << "Error in compute radiation." << std::endl << std::endl;

    //SAVE OUTPUT
    std::string otputFileName;
    otputFileName = path.toStdString() + "globalRadiation";
    if (gis::writeEsriGrid(otputFileName, radMaps->globalRadiationMap, error))
        std::cout << "Map saved in: " << otputFileName << std::endl;
    else
        std::cout << "Error in writing:" << otputFileName << std::endl << *error << std::endl;

    return a.exec();
}

