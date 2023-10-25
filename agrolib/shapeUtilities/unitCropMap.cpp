#include "commonConstants.h"
#include "unitCropMap.h"
#include "zonalStatistic.h"
#include "shapeToRaster.h"
#include "shapeUtilities.h"
#include "formInfo.h"
#include "computationUnitsDb.h"

#include <QFile>


bool computeUcmPrevailing(Crit3DShapeHandler &shapeUCM, Crit3DShapeHandler &shapeCrop, Crit3DShapeHandler &shapeSoil, Crit3DShapeHandler &shapeMeteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize, double threshold,
                 QString ucmFileName, std::string &error, bool showInfo)
{

    // make a copy of crop shapefile (reference) and return cloned shapefile complete path
    QString refFileName = QString::fromStdString(shapeCrop.getFilepath());
    QString ucmShapeFileName = cloneShapeFile(refFileName, ucmFileName);

    if (!shapeUCM.open(ucmShapeFileName.toStdString()))
    {
        error = "Load shapefile failed: " + ucmShapeFileName.toStdString();
        return false;
    }

    // create reference and value raster
    gis::Crit3DRasterGrid rasterRef;
    gis::Crit3DRasterGrid rasterVal;
    initializeRasterFromShape(shapeUCM, rasterRef, cellSize);
    initializeRasterFromShape(shapeUCM, rasterVal, cellSize);

    FormInfo formInfo;

    // CROP (reference shape)
    if (showInfo) formInfo.start("[1/8] Rasterize crop (reference)...", 0);
    fillRasterWithShapeNumber(rasterRef, shapeUCM);

    // meteo grid
    if (showInfo) formInfo.setText("[2/8] Rasterize meteo grid...");
    fillRasterWithShapeNumber(rasterVal, shapeMeteo);

    if (showInfo) formInfo.setText("[3/8] Compute matrix crop/meteo...");
    std::vector <int> vectorNull;
    std::vector <std::vector<int> > matrix = computeMatrixAnalysis(shapeUCM, shapeMeteo, rasterRef, rasterVal, vectorNull);

    if (showInfo) formInfo.setText("[4/8] Zonal statistic crop/meteo...");
    bool isOk = zonalStatisticsShapeMajority(shapeUCM, shapeMeteo, matrix, vectorNull, idMeteo, "ID_METEO", threshold, error);

    // zonal statistic on soil map
    if (isOk)
    {
        if (showInfo) formInfo.setText("[5/8] Rasterize soil...");
        fillRasterWithShapeNumber(rasterVal, shapeSoil);

        if (showInfo) formInfo.setText("[6/8] Compute matrix crop/soil...");
        matrix = computeMatrixAnalysis(shapeUCM, shapeSoil, rasterRef, rasterVal, vectorNull);

        if (showInfo) formInfo.setText("[7/8] Zonal statistic crop/soil...");
        isOk = zonalStatisticsShapeMajority(shapeUCM, shapeSoil, matrix, vectorNull, idSoil, "ID_SOIL", threshold, error);
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

    if (showInfo) formInfo.setText("[8/8] Write UCM...");

    // add ID CASE
    shapeUCM.addField("ID_CASE", FTString, 20, 0);
    int idCaseIndex = shapeUCM.getFieldPos("ID_CASE");

    // add ID CROP
    bool existIdCrop = shapeUCM.existField("ID_CROP");
    if (! existIdCrop) shapeUCM.addField("ID_CROP", FTString, 5, 0);
    int idCropIndex = shapeUCM.getFieldPos("ID_CROP");

    // read indexes
    int nShape = shapeUCM.getShapeCount();
    int cropIndex = shapeUCM.getFieldPos(idCrop);
    if(cropIndex == -1)
    {
        error = "Missing idCrop: " + idCrop;
        return false;
    }
    int soilIndex = shapeUCM.getFieldPos("ID_SOIL");
    if(soilIndex == -1)
    {
        error = "Missing idSoil: " + idSoil;
        return false;
    }
    int meteoIndex = shapeUCM.getFieldPos("ID_METEO");
    if(meteoIndex == -1)
    {
        error = "Missing idMeteo: " + idMeteo;
        return false;
    }

    // FILL ID_CROP and ID_CASE
    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        std::string cropStr = shapeUCM.readStringAttribute(shapeIndex, cropIndex);
        if (cropStr == "-9999") cropStr = "";

        std::string soilStr = shapeUCM.readStringAttribute(shapeIndex, soilIndex);
        if (soilStr == "-9999") soilStr = "";

        std::string meteoStr = shapeUCM.readStringAttribute(shapeIndex, meteoIndex);
        if (meteoStr == "-9999") meteoStr = "";

        std::string caseStr = "";
        if (meteoStr != "" && soilStr != "" && cropStr != "")
            caseStr = "M" + meteoStr + "S" + soilStr + "C" + cropStr;

        if (! existIdCrop) shapeUCM.writeStringAttribute(shapeIndex, idCropIndex, cropStr.c_str());
        shapeUCM.writeStringAttribute(shapeIndex, idCaseIndex, caseStr.c_str());

        if (caseStr == "")
            shapeUCM.deleteRecord(shapeIndex);
    }

    if (showInfo) formInfo.close();
    cleanShapeFile(shapeUCM);

    return isOk;
}


bool fillUcmIdCase(Crit3DShapeHandler &ucm, std::string idCrop, std::string idSoil, std::string idMeteo)
{
    if (!ucm.existField("ID_CASE"))
    {
        return false;
    }
    // read indexes
    int nShape = ucm.getShapeCount();
    int cropIndex = ucm.getFieldPos(idCrop);
    int soilIndex = ucm.getFieldPos(idSoil);
    int meteoIndex = ucm.getFieldPos(idMeteo);
    int idCaseIndex = ucm.getFieldPos("ID_CASE");

    if (cropIndex == -1 || soilIndex == -1 || meteoIndex == -1)
    {
        return false;
    }
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

        ucm.writeStringAttribute(shapeIndex, idCaseIndex, caseStr.c_str());

        if (caseStr == "")
            ucm.deleteRecord(shapeIndex);
    }
    return true;
}


bool writeUcmListToDb(Crit3DShapeHandler &shapeHandler, QString dbName, QString &error)
{
    int nrShape = shapeHandler.getShapeCount();
    if (nrShape <= 0)
    {
        error = "Shapefile is void.";
        return false;
    }

    QList<QString> idCase, idCrop, idMeteo, idSoil;
    QList<double> ha;

    for (int i = 0; i < nrShape; i++)
    {
        QString key = QString::fromStdString(shapeHandler.getStringValue(signed(i), "ID_CASE"));
        if (key.isEmpty()) continue;

        double hectares = shapeHandler.getNumericValue(signed(i), "HA");
        if (hectares == NODATA)
            hectares = shapeHandler.getNumericValue(signed(i), "Hectares");
        if (hectares == NODATA)
            hectares = shapeHandler.getNumericValue(signed(i), "hectares");

        if ( !idCase.contains(key) )
        {
            idCase << key;
            idCrop << QString::fromStdString(shapeHandler.getStringValue(signed(i), "ID_CROP"));
            idMeteo << QString::fromStdString(shapeHandler.getStringValue(signed(i), "ID_METEO"));
            idSoil << QString::fromStdString(shapeHandler.getStringValue(signed(i), "ID_SOIL"));
            ha << hectares;
        }
        else
        {
            // sum hectares
            if (hectares != NODATA && hectares > 0)
            {
                int index = idCase.indexOf(key);
                ha[index] += hectares;
            }
        }
    }

    ComputationUnitsDB compUnitsDb(dbName, error);
    if (error != "")
        return false;

    return compUnitsDb.writeListToCompUnitsTable(idCase, idCrop, idMeteo, idSoil, ha, error);

}
