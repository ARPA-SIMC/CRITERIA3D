#include "unitCropMap.h"
#include "zonalStatistic.h"
#include "shapeToRaster.h"
#include "shapeUtilities.h"

#include <QFile>
#include <QFileInfo>
#include <qdebug.h>


bool computeUcmPrevailing(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize,
                 QString ucmFileName, std::string *error, bool showInfo)
{

    // make a copy of shapefile and return cloned shapefile complete path
    QString refFileName = QString::fromStdString(crop->getFilepath());
    QString ucmShapeFileName = cloneShapeFile(refFileName, ucmFileName);

    if (!ucm->open(ucmShapeFileName.toStdString()))
    {
        *error = "Load shapefile failed: " + ucmShapeFileName.toStdString();
        return false;
    }

    // create reference and value raster
    gis::Crit3DRasterGrid* rasterRef = new(gis::Crit3DRasterGrid);
    gis::Crit3DRasterGrid* rasterVal = new(gis::Crit3DRasterGrid);
    initializeRasterFromShape(ucm, rasterRef, cellSize);
    initializeRasterFromShape(ucm, rasterVal, cellSize);

    // ECM --> reference
    fillRasterWithShapeNumber(rasterRef, ucm, showInfo);

    // zonal statistic on meteo grid
    fillRasterWithShapeNumber(rasterVal, meteo, showInfo);
    bool isOk = zonalStatisticsShape(ucm, meteo, rasterRef, rasterVal, idMeteo, "ID_METEO", MAJORITY, error, showInfo);

    // zonal statistic on soil map
    if (isOk)
    {
        fillRasterWithShapeNumber(rasterVal, soil, showInfo);
        isOk = zonalStatisticsShape(ucm, soil, rasterRef, rasterVal, idSoil, "ID_SOIL", MAJORITY, error, showInfo);
    }

    if (! isOk)
    {
        *error = "ZonalStatisticsShape: " + *error;
    }

    delete rasterRef;
    delete rasterVal;
    if (! isOk) return false;

    // add ID CASE
    ucm->addField("ID_CASE", FTString, 20, 0);
    int idCaseIndex = ucm->getFieldPos("ID_CASE");

    // add ID CROP
    bool existIdCrop = ucm->existField("ID_CROP");
    if (! existIdCrop) ucm->addField("ID_CROP", FTString, 5, 0);
    int idCropIndex = ucm->getFieldPos("ID_CROP");

    // read indexes
    int nShape = ucm->getShapeCount();
    int cropIndex = ucm->getFieldPos(idCrop);
    int soilIndex = ucm->getFieldPos(idSoil);
    int meteoIndex = ucm->getFieldPos(idMeteo);

    // FILL ID_CROP and ID_CASE
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

        if (! existIdCrop) ucm->writeStringAttribute(shapeIndex, idCropIndex, cropStr.c_str());
        ucm->writeStringAttribute(shapeIndex, idCaseIndex, caseStr.c_str());

        if (caseStr == "")
            ucm->deleteRecord(shapeIndex);
    }

    cleanShapeFile(ucm);

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


bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, QString ucmFileName, std::string *error, bool showInfo)
{

    // PolygonShapefile
    int type = 5;

    ucm->newShapeFile(ucmFileName.toStdString(), type);
    // copy .prj
    QFileInfo refFileInfo;
    if (crop != nullptr)
    {
        refFileInfo.setFile(QString::fromStdString(crop->getFilepath()));
    }
    else if(soil!=nullptr)
    {
        refFileInfo.setFile(QString::fromStdString(soil->getFilepath()));
    }
    QString refFile = refFileInfo.absolutePath() + "/" + refFileInfo.baseName();
    QFileInfo ucmFileInfo(ucmFileName);
    QString ucmFile = ucmFileInfo.absolutePath() + "/" + ucmFileInfo.baseName();
    QFile::copy(refFile +".prj", ucmFile +".prj");

    ucm->open(ucmFileName.toStdString());
    // add ID CASE
    ucm->addField("ID_CASE", FTString, 20, 0);
    // add ID SOIL
    ucm->addField("ID_SOIL", FTString, 5, 0);
    int soilIndex = ucm->getFieldPos("ID_SOIL");
    // add ID CROP
    ucm->addField("ID_CROP", FTString, 5, 0);
    int cropIndex = ucm->getFieldPos("ID_CROP");
    // add ID METEO
    ucm->addField("ID_METEO", FTString, 5, 0);
    int meteoIndex = ucm->getFieldPos("ID_METEO");

    qDebug() << "idCrop " << QString::fromStdString(idCrop);
    qDebug() << "idSoil " << QString::fromStdString(idSoil);
    qDebug() << "idMeteo " << QString::fromStdString(idMeteo);

    #ifdef GDAL
    GEOSGeometry *inteserctionGeom = nullptr ;

    if (crop == nullptr)
    {

        // soil and meteo intersection, add constant idCrop
        if (!shapeIntersection(soil, meteo, &inteserctionGeom))
        {
            return false;
        }

    }
    else if (soil == nullptr)
    {

        // crop and meteo intersection, add constant idSoil
        if (!shapeIntersection(crop, meteo, &inteserctionGeom))
        {
            return false;
        }
    }
    else if (meteo == nullptr)
    {

        // crop and soil intersection, add constant idMeteo
        if (!shapeIntersection(crop, soil, &inteserctionGeom))
        {
            return false;
        }
    }
    else
    {
        // TO DO
    }

    if (!getShapeFromGeom(inteserctionGeom, ucm))
    {
        return false;
    }

    // Finalizzo GEOS
    finishGEOS();
    #endif //GDAL

    /*
    int nShape = ucm->getShapeCount();
    for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
    {
        ucm->writeStringAttribute(shapeIndex, soilIndex, idSoil.c_str());
        ucm->writeStringAttribute(shapeIndex, cropIndex, idCrop.c_str());
        ucm->writeStringAttribute(shapeIndex, meteoIndex, idMeteo.c_str());
    }
    */

    /*
    if (!fillIDCase(ucm, idCrop, idSoil, idMeteo))
    {
        *error = "Failed to fill ID CASE";
        return false;
    }
    */

    ucm->close();
    ucm->open(ucm->getFilepath());
    return true;
}


#ifdef GDAL
bool shapeIntersection(Crit3DShapeHandler *first, Crit3DShapeHandler *second, GEOSGeometry **inteserctionGeom)
{
    GEOSGeometry* firstPolygon = loadShapeAsPolygon(first);
    if((GEOSisEmpty(firstPolygon)))
    {
        qDebug() << "cropPolygon empty";
        return false;
    }

    if (GEOSisValid(firstPolygon) !=1)
    {
          qDebug() << "firstPolygon is NOT Valid";
          qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(firstPolygon);
          firstPolygon = GEOSMakeValid(firstPolygon);
          qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(firstPolygon);
    }
   else
      qDebug() << "firstPolygon is Valid";

    GEOSGeometry *secondPolygon = loadShapeAsPolygon(second);
    if((GEOSisEmpty(secondPolygon)))
    {
        qDebug() << "secondPolygon empty";
        return false;
    }

    if (GEOSisValid(secondPolygon) !=1)
    {
          qDebug() << "secondPolygon is NOT Valid";
          qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(secondPolygon);
          secondPolygon = GEOSMakeValid(secondPolygon);
          qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(secondPolygon);
    }
   else
      qDebug() << "soilPolygon is Valid";

    *inteserctionGeom = GEOSIntersection(firstPolygon, secondPolygon);
    if ((*inteserctionGeom) == nullptr)
    {
        qDebug() << "inteserctionGeom nullptr";
        return false;
    }
    if((GEOSisEmpty(*inteserctionGeom)))
    {
        qDebug() << "inteserctionGeom empty";
        return false;
    }

    if (GEOSisValid(*inteserctionGeom) !=1)
    {
          qDebug() << "inteserctionGeom is NOT Valid";
          return false;
    }
   else
    {
      qDebug() << "inteserctionGeom is Valid";
      qDebug() << "Resulting geometry is " << GEOSGeomToWKT(*inteserctionGeom);
      return true;
    }
}

bool getShapeFromGeom(GEOSGeometry *inteserctionGeom, Crit3DShapeHandler *ucm)
{
    //Getting coords for the vertex
    unsigned int num;
    int numPoints;

    GEOSGeom geom;
    num = GEOSGetNumGeometries(inteserctionGeom);
    qDebug () << "Geometries: " << num;

    GEOSCoordSeq coordseqIntersection = nullptr;
    const GEOSGeometry *ring;
    coordseqIntersection = (GEOSCoordSeq) GEOSCoordSeq_create(2, 2);   //2 pointsbi-dimensional
    std::vector<double> coordinates;
    std::string type;

    int nValidShape = 0;
    for(int i=0; i < num; i++)
    {
        coordinates.clear();
        geom = (GEOSGeom) GEOSGetGeometryN(inteserctionGeom, i);
        type = GEOSGeomType(geom);
        if (type != "Polygon")
        {
            continue;
        }
        ring = GEOSGetExteriorRing(geom);

        if (ring)
        {
            numPoints = GEOSGeomGetNumPoints(ring);
            coordseqIntersection = (GEOSCoordSeq) GEOSGeom_getCoordSeq(ring);
            double xPoint;
            double yPoint;

            for (int p=0; p < numPoints; p++)
            {

                GEOSCoordSeq_getX(coordseqIntersection, p, &xPoint);
                GEOSCoordSeq_getY(coordseqIntersection, p, &yPoint);

                coordinates.push_back(xPoint);
                coordinates.push_back(yPoint);
            }
            qDebug () << "GEOSGetNumInteriorRings( geom ) " << GEOSGetNumInteriorRings( geom );

            //interior rings TBC
            for ( int numInner = 0; numInner < GEOSGetNumInteriorRings( geom ); numInner++ )
            {
                ring = GEOSGetInteriorRingN( geom, numInner );
                numPoints = GEOSGeomGetNumPoints(ring);
                coordseqIntersection = (GEOSCoordSeq) GEOSGeom_getCoordSeq(ring);

                 for ( unsigned int j = 0; j < numPoints; j++ )
                 {
                     GEOSCoordSeq_getX(coordseqIntersection, j, &xPoint);
                     GEOSCoordSeq_getY(coordseqIntersection, j, &yPoint);

                     coordinates.push_back(xPoint);
                     coordinates.push_back(yPoint);
                 }

            }
            if (ucm->addShape(nValidShape, type, coordinates))
            {
                nValidShape = nValidShape + 1;
            }
        }
    }
    return true;
}

#endif // GDAL


