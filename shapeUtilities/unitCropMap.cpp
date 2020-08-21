#include "unitCropMap.h"
#include "zonalStatistic.h"
#include "shapeToRaster.h"
#include "shapeUtilities.h"
#include "formInfo.h"

#include <QFile>
#include <QFileInfo>
#include <qdebug.h>


bool computeUcmPrevailing(Crit3DShapeHandler &ucm, Crit3DShapeHandler &crop, Crit3DShapeHandler &soil, Crit3DShapeHandler &meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize,
                 QString ucmFileName, std::string &error, bool showInfo)
{

    // make a copy of shapefile and return cloned shapefile complete path
    QString refFileName = QString::fromStdString(crop.getFilepath());
    QString ucmShapeFileName = cloneShapeFile(refFileName, ucmFileName);

    if (!ucm.open(ucmShapeFileName.toStdString()))
    {
        error = "Load shapefile failed: " + ucmShapeFileName.toStdString();
        return false;
    }

    // create reference and value raster
    gis::Crit3DRasterGrid rasterRef;
    gis::Crit3DRasterGrid rasterVal;
    initializeRasterFromShape(&ucm, &rasterRef, cellSize);
    initializeRasterFromShape(&ucm, &rasterVal, cellSize);

    FormInfo formInfo;

    // CROP --> reference
    if (showInfo) formInfo.start("Rasterize crop...", 0);
    fillRasterWithShapeNumber(&rasterRef, &ucm);

    // meteo grid
    if (showInfo) formInfo.start("Rasterize meteo grid...", 0);
    fillRasterWithShapeNumber(&rasterVal, &meteo);

    if (showInfo) formInfo.start("Compute matrix...", 0);
    std::vector <int> vectorNull;
    std::vector <std::vector<int> > matrix = computeMatrixAnalysis(ucm, meteo, rasterRef, rasterVal, vectorNull);

    if (showInfo) formInfo.start("Zonal statistic...", 0);
    bool isOk = zonalStatisticsShapeMajority(ucm, meteo, matrix, vectorNull, idMeteo, "ID_METEO", error);

    // zonal statistic on soil map
    if (isOk)
    {
        if (showInfo) formInfo.start("Rasterize soil...", 0);
        fillRasterWithShapeNumber(&rasterVal, &soil);

        if (showInfo) formInfo.start("Compute matrix...", 0);
        matrix = computeMatrixAnalysis(ucm, meteo, rasterRef, rasterVal, vectorNull);

        if (showInfo) formInfo.start("Zonal statistic...", 0);
        isOk = zonalStatisticsShapeMajority(ucm, soil, matrix, vectorNull, idSoil, "ID_SOIL", error);
    }

    if (! isOk)
    {
        error = "ZonalStatisticsShape: " + error;
    }

    rasterRef.clear();
    rasterVal.clear();
    matrix.clear();
    vectorNull.clear();

    if (! isOk)
    {
        if (showInfo) formInfo.close();
        return false;
    }

    if (showInfo) formInfo.start("Write UCM...", 0);

    // add ID CASE
    ucm.addField("ID_CASE", FTString, 20, 0);
    int idCaseIndex = ucm.getFieldPos("ID_CASE");

    // add ID CROP
    bool existIdCrop = ucm.existField("ID_CROP");
    if (! existIdCrop) ucm.addField("ID_CROP", FTString, 5, 0);
    int idCropIndex = ucm.getFieldPos("ID_CROP");

    // read indexes
    int nShape = ucm.getShapeCount();
    int cropIndex = ucm.getFieldPos(idCrop);
    int soilIndex = ucm.getFieldPos(idSoil);
    int meteoIndex = ucm.getFieldPos(idMeteo);

    // FILL ID_CROP and ID_CASE
    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        std::string cropStr = ucm.readStringAttribute(shapeIndex, cropIndex);
        if (cropStr == "-9999") cropStr = "";

        std::string soilStr = ucm.readStringAttribute(shapeIndex, soilIndex);
        if (soilStr == "-9999") soilStr = "";

        std::string meteoStr = ucm.readStringAttribute(shapeIndex, meteoIndex);
        if (meteoStr == "-9999") meteoStr = "";

        std::string caseStr = "";
        if (meteoStr != "" && soilStr != "" && cropStr != "")
            caseStr = "M" + meteoStr + "S" + soilStr + "C" + cropStr;

        if (! existIdCrop) ucm.writeStringAttribute(shapeIndex, idCropIndex, cropStr.c_str());
        ucm.writeStringAttribute(shapeIndex, idCaseIndex, caseStr.c_str());

        if (caseStr == "")
            ucm.deleteRecord(shapeIndex);
    }

    if (showInfo) formInfo.close();
    cleanShapeFile(&ucm);

    return isOk;
}


// FILL ID_CASE
bool fillIDCase(Crit3DShapeHandler *ucm, std::string idCrop, std::string idSoil, std::string idMeteo)
{
    if (!ucm->existField("ID_CASE"))
    {
        return false;
    }
    // read indexes
    int nShape = ucm->getShapeCount();
    int cropIndex = ucm->getFieldPos(idCrop);
    int soilIndex = ucm->getFieldPos(idSoil);
    int meteoIndex = ucm->getFieldPos(idMeteo);
    int idCaseIndex = ucm->getFieldPos("ID_CASE");

    if (cropIndex == -1 || soilIndex == -1 || meteoIndex == -1)
    {
        return false;
    }
    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        std::string cropStr = ucm->readStringAttribute(shapeIndex, cropIndex);
        if (cropStr == "-9999") cropStr = "";

        std::string soilStr = ucm->readStringAttribute(shapeIndex, soilIndex);
        if (soilStr == "-9999") soilStr = "";

        std::string meteoStr = ucm->readStringAttribute(shapeIndex, meteoIndex);
        if (meteoStr == "-9999") meteoStr = "";

        std::string caseStr = "";
        if (meteoStr != "" && soilStr != "" && cropStr != "")
            caseStr = "M" + meteoStr + "S" + soilStr + "C" + cropStr;

        ucm->writeStringAttribute(shapeIndex, idCaseIndex, caseStr.c_str());

        if (caseStr == "")
            ucm->deleteRecord(shapeIndex);
    }
    return true;
}
