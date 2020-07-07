#include "unitCropMap.h"
#include "zonalStatistic.h"
#include "shapeToRaster.h"
#include "shapeUtilities.h"
#include <QFile>
#include <QFileInfo>
#include <QPolygon>
//#include <geos/operation/valid/MakeValid.h>

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

bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, QString ucmFileName, std::string *error, bool showInfo)
{

    // PolygonShapefile
    int type = 2;

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

    int nShape = ucm->getShapeCount();

    qDebug() << "idCrop " << QString::fromStdString(idCrop);
    qDebug() << "idSoil " << QString::fromStdString(idSoil);
    qDebug() << "idMeteo " << QString::fromStdString(idMeteo);

    //testIntersection();
    //return true;
    if (crop == nullptr)
    {

        // soil and meteo intersection, add constant idCrop
        GEOSGeometry* soilPolygon = loadShapeAsPolygon(soil);
        if ( GEOSisValid(soilPolygon) )
              qDebug() << "soilPolygon isValid";
           else
              qDebug() << "soilPolygon is NOT Valid";

        GEOSGeometry *meteoPolygon = loadShapeAsPolygon(meteo);
        if ( GEOSisValid(meteoPolygon) )
              qDebug() << "meteoPolygon isValid";
           else
              qDebug() << "meteoPolygon is NOT Valid";

        if(soilPolygon == NULL || meteoPolygon == NULL) {
            qDebug() << "NULL polygon";
            return false;    //invalid input parameter
        }
        GEOSGeometry *inteserctionGeom = GEOSIntersection(soilPolygon, meteoPolygon);
        if ( !inteserctionGeom )
        {
            if ( GEOSisValid(inteserctionGeom) )
                  qDebug() << "inteserctionGeom isValid";
               else
                  qDebug() << "inteserctionGeom is NOT Valid";
            qDebug() << "Resulting geometry is " << GEOSGeomToWKT(inteserctionGeom);
        }
        for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
        {
            ucm->writeStringAttribute(shapeIndex, cropIndex, idCrop.c_str());
        }
    }
    else if (soil == nullptr)
    {

        // crop and meteo intersection, add constant idSoil
        GEOSGeometry* cropPolygon = loadShapeAsPolygon(crop);
        if ( GEOSisValid(cropPolygon) )
              qDebug() << "cropPolygon isValid";
           else
              qDebug() << "cropPolygon is NOT Valid";

        GEOSGeometry *meteoPolygon = loadShapeAsPolygon(meteo);
        if ( GEOSisValid(meteoPolygon) )
              qDebug() << "meteoPolygon isValid";
           else
              qDebug() << "meteoPolygon is NOT Valid";

        if(cropPolygon == NULL || meteoPolygon == NULL) {
            qDebug() << "NULL polygon";
            return false;    //invalid input parameter
        }
        GEOSGeometry *inteserctionGeom = GEOSIntersection(cropPolygon, meteoPolygon);
        if ( !inteserctionGeom )
        {
            if ( GEOSisValid(inteserctionGeom) )
                  qDebug() << "inteserctionGeom isValid";
               else
                  qDebug() << "inteserctionGeom is NOT Valid";
            qDebug() << "Resulting geometry is " << GEOSGeomToWKT(inteserctionGeom);
        }
        for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
        {
            ucm->writeStringAttribute(shapeIndex, soilIndex, idSoil.c_str());
        }
    }
    else if (meteo == nullptr)
    {

        // crop and soil intersection, add constant idMeteo
        GEOSGeometry* cropPolygon = loadShapeAsPolygon(crop);
        if((GEOSisEmpty(cropPolygon)))
        {
            qDebug() << "cropPolygon empty";
            //return false;
        }

        if (GEOSisValid(cropPolygon) !=1)
        {
              qDebug() << "cropPolygon is NOT Valid";
              qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(cropPolygon);
              cropPolygon = GEOSMakeValid(cropPolygon);
              qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(cropPolygon);
              //return false;
        }
       else
          qDebug() << "cropPolygon is Valid";

        GEOSGeometry *soilPolygon = loadShapeAsPolygon(soil);
        if((GEOSisEmpty(soilPolygon)))
        {
            qDebug() << "soilPolygon empty";
            //return false;
        }

        if (GEOSisValid(soilPolygon) !=1)
        {
              qDebug() << "soilPolygon is NOT Valid";
              qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(soilPolygon);
              soilPolygon = GEOSMakeValid(soilPolygon);
              qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(soilPolygon);
              //return false;
        }
       else
          qDebug() << "soilPolygon is Valid";

        GEOSGeometry *inteserctionGeom = GEOSIntersection(cropPolygon, soilPolygon);
        if (inteserctionGeom == NULL)
        {
            qDebug() << "inteserctionGeom NULL";
            return false;
        }
        if((GEOSisEmpty(inteserctionGeom)))
        {
            qDebug() << "inteserctionGeom empty";
            //return false;
        }

        if (GEOSisValid(inteserctionGeom) !=1)
        {
              qDebug() << "inteserctionGeom is NOT Valid";
              //return false;
        }
       else
        {
          qDebug() << "inteserctionGeom is Valid";
          qDebug() << "Resulting geometry is " << GEOSGeomToWKT(inteserctionGeom);
        }

        //Getting coords for the vertex
        unsigned int num;
        int numPoints;

        GEOSGeom geom;
        num = GEOSGetNumGeometries(inteserctionGeom);
        printf("Geometries: %d\n",num);

        GEOSCoordSeq coordseqIntersection = NULL;
        coordseqIntersection = (GEOSCoordSeq) GEOSCoordSeq_create(2, 2);   //2 pointsbi-dimensional
        std::vector<double> coordinates;

        for(int i=0; i < num; i++)
        {
            coordinates.clear();
            geom = (GEOSGeom) GEOSGetGeometryN(inteserctionGeom, i);
qDebug() << GEOSGeomType(geom);
            coordseqIntersection = (GEOSCoordSeq) GEOSGeom_getCoordSeq(geom);

            numPoints = GEOSGeomGetNumPoints(geom);
            for (int p=0; p < numPoints; p++)
            {
                double xPoint;
                double yPoint;

                GEOSCoordSeq_getX(coordseqIntersection, 0, &xPoint);
                GEOSCoordSeq_getY(coordseqIntersection, 0, &yPoint);

                coordinates.push_back(xPoint);
                coordinates.push_back(yPoint);
            }
            if (!ucm->addShape(i, coordinates))
            {
                return false;
            }
        }

        // Finalizzo GEOS
        finishGEOS();
        /*
        Crit3DShapeHandler *temp = new(Crit3DShapeHandler);
        temp->newShapeFile("temp", type);
        temp->open("temp");
        // add ID SOIL
        temp->addField("ID_SOIL", FTString, 5, 0);
        // add ID CROP
        temp->addField("ID_ECM", FTString, 5, 0);
        // soil and meteo intersection, shape result and crop intersection
        if (!shapeIntersection(temp, soil, crop, idSoil, idCrop, error, showInfo))
        {
            *error = "Failed soil and meteo intersection";
            delete temp;
            return false;
        }
        */
        for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
        {
            ucm->writeStringAttribute(shapeIndex, meteoIndex, idMeteo.c_str());
        }
    }
    else
    {
        /*
        Crit3DShapeHandler *temp = new(Crit3DShapeHandler);
        temp->newFile("temp", type);
        temp->open("temp");
        // add ID SOIL
        temp->addField("ID_SOIL", FTString, 5, 0);
        // add ID METEO
        temp->addField("ID_METEO", FTString, 5, 0);
        // soil and meteo intersection, shape result and crop intersection
        if (!shapeIntersection(temp, soil, meteo, idSoil, idMeteo, error, showInfo))
        {
            *error = "Failed soil and meteo intersection";
            delete temp;
            return false;
        }
        if (!shapeIntersection(ucm, temp, crop, idSoil, idCrop, error, showInfo))
        {
            *error = "Failed crop intersection";
            return false;
        }
        */

    }
    if (!fillIDCase(ucm, idCrop, idSoil, idMeteo))
    {
        *error = "Failed to fill ID CASE";
        return false;
    }
    return true;
}


bool shapeIntersection(Crit3DShapeHandler *intersecHandler, Crit3DShapeHandler *firstHandler, Crit3DShapeHandler *secondHandler, std::string fieldNameFirst, std::string fieldNameSecond, std::string *error, bool showInfo)
{
    ShapeObject myFirstShape;
    ShapeObject mySecondShape;
    Box<double> firstBounds;
    Box<double> secondBounds;
    int nrFirstShape = firstHandler->getShapeCount();
    std::vector<ShapeObject::Part> shapeFirstParts;
    std::vector<ShapeObject::Part> shapeSecondParts;
    int nrSecondShape = secondHandler->getShapeCount();
    int nIntersections = 0;
    std::vector< std::vector<ShapeObject::Part>> firstShapeParts;
    std::vector< std::vector<ShapeObject::Part>> secondShapeParts;

    QPolygonF firstPolygon;
    QPolygonF firstHolePolygon;
    QPolygonF secondHolePolygon;
    QPolygonF secondPolygon;
    QPolygonF intersectionPolygon;

    int IDfirstShape = firstHandler->getFieldPos(fieldNameFirst);
    int IDsecondShape = secondHandler->getFieldPos(fieldNameSecond);
    int IDCloneFirst = intersecHandler->getFieldPos(fieldNameFirst);
    int IDCloneSecond = intersecHandler->getFieldPos(fieldNameSecond);
    int iShape = 0;

    for (unsigned int firstShapeIndex = 0; firstShapeIndex < nrFirstShape; firstShapeIndex++)
    {

        firstPolygon.clear();
        firstHandler->getShape(firstShapeIndex, myFirstShape);
        shapeFirstParts = myFirstShape.getParts();
        std::string fieldFirst = firstHandler->readStringAttribute(firstShapeIndex, IDfirstShape);  //Field to copy
        // get bounds
        firstBounds = myFirstShape.getBounds();
        firstShapeParts[firstShapeIndex] = myFirstShape.getParts();
        for (unsigned int partIndex = 0; partIndex < firstShapeParts[firstShapeIndex].size(); partIndex++)
        {
            Box<double> partBB = myFirstShape.getPart(partIndex).boundsPart;
            int offset = myFirstShape.getPart(partIndex).offset;
            int length = myFirstShape.getPart(partIndex).length;
            std::vector<unsigned int> holesFirstParts = firstHandler->getHoles(firstShapeIndex,partIndex);

            if (firstShapeParts[firstShapeIndex][partIndex].hole)
            {
                continue;
            }
            else
            {
                for (unsigned long v = 0; v < length; v++)
                {
                    Point<double> vertex = myFirstShape.getVertex(v+offset);
                    QPoint point(vertex.x, vertex.y);
                    firstPolygon.append(point);
                }
                // check holes
                for (int holesIndex = 0; holesIndex < holesFirstParts.size(); holesIndex++)
                {
                    firstHolePolygon.clear();
                    int offset = myFirstShape.getPart(holesFirstParts[holesIndex]).offset;
                    int length = myFirstShape.getPart(holesFirstParts[holesIndex]).length;
                    std::vector<double> x;
                    std::vector<double> y;
                    for (unsigned long v = 0; v < length; v++)
                    {
                        Point<double> holeVertex = myFirstShape.getVertex(v+offset);
                        QPoint point(holeVertex.x, holeVertex.y);
                        firstHolePolygon.append(point);
                    }
                    firstPolygon = firstPolygon.subtracted(firstHolePolygon);
                }
            }
        }
        for (unsigned int secondShapeIndex = 0; secondShapeIndex < nrSecondShape; secondShapeIndex++)
        {

            secondPolygon.clear();
            secondHandler->getShape(secondShapeIndex, mySecondShape);
            std::string fieldSecond = secondHandler->readStringAttribute(secondShapeIndex, IDsecondShape); //Field to copy
            // get bounds
            secondBounds = mySecondShape.getBounds();
            bool noOverlap = firstBounds.xmin > secondBounds.xmax ||
                                 secondBounds.xmin > firstBounds.xmax ||
                                 firstBounds.ymin > secondBounds.ymax ||
                                 secondBounds.ymin > firstBounds.ymax;
            if (noOverlap)
            {
                continue;
            }
            else
            {
                secondShapeParts[secondShapeIndex] = mySecondShape.getParts();
                for (unsigned int partIndex = 0; partIndex < secondShapeParts[secondShapeIndex].size(); partIndex++)
                {
                    Box<double> partBB = mySecondShape.getPart(partIndex).boundsPart;
                    int offset = mySecondShape.getPart(partIndex).offset;
                    int length = mySecondShape.getPart(partIndex).length;

                    std::vector<unsigned int> holesSecondParts = secondHandler->getHoles(secondShapeIndex,partIndex);
                    if (secondShapeParts[secondShapeIndex][partIndex].hole)
                    {
                        continue;
                    }
                    else
                    {
                        nIntersections = nIntersections + 1;
                        for (unsigned long v = 0; v < length; v++)
                        {
                            Point<double> vertex = mySecondShape.getVertex(v+offset);
                            QPoint point(vertex.x, vertex.y);
                            secondPolygon.append(point);
                        }
                        // check holes
                        for (int holesIndex = 0; holesIndex < holesSecondParts.size(); holesIndex++)
                        {
                            secondHolePolygon.clear();
                            int offset = mySecondShape.getPart(holesSecondParts[holesIndex]).offset;
                            int length = mySecondShape.getPart(holesSecondParts[holesIndex]).length;
                            std::vector<double> x;
                            std::vector<double> y;
                            for (unsigned long v = 0; v < length; v++)
                            {
                                Point<double> holeVertex = mySecondShape.getVertex(v+offset);
                                QPoint point(holeVertex.x, holeVertex.y);
                                secondHolePolygon.append(point);
                            }
                            secondPolygon = secondPolygon.subtracted(secondHolePolygon);
                        }
                    }
                }
                intersectionPolygon = firstPolygon.intersected(secondPolygon);
                std::vector<double> coordinates;
                for (int i = 0; i<intersectionPolygon.size(); i++)
                {
                    coordinates.push_back(intersectionPolygon[i].x());
                    coordinates.push_back(intersectionPolygon[i].y());
                }
                if (!intersecHandler->addShape(iShape, coordinates))
                {
                    return false;
                }
                iShape = iShape + 1;
                if (!intersecHandler->writeStringAttribute(nIntersections, IDCloneFirst, fieldFirst.c_str()))
                {
                    return false;
                }
                if (!intersecHandler->writeStringAttribute(nIntersections, IDCloneSecond, fieldSecond.c_str()))
                {
                    return false;
                }
            }
        }
    }

    return true;
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



