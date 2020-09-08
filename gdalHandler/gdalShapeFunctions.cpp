#include "gdalShapeFunctions.h"
#include <QFileInfo>
#include <string.h>
#include <qdebug.h>
#include <ogrsf_frmts.h>
#include "ogr_spatialref.h"
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <gdalwarper.h>


bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, QString ucmFileName, std::string *error)
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


GEOSGeometry * loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler)
{
    // Init GEOS
    GEOSMessageHandler error_function = nullptr, notice_function = nullptr;
    initGEOS(notice_function, error_function);

    ShapeObject shapeObj;

    int nShapes = shapeHandler->getShapeCount();
    std::vector<ShapeObject::Part> shapeParts;

    QVector<GEOSGeometry *> geometries;
    QVector<GEOSGeometry *> holes;

    std::vector<double> xVertex;
    std::vector<double> yVertex;
    std::vector<std::vector <double> > xVertexHoles;
    std::vector<std::vector <double> > yVertexHoles;

    GEOSCoordSequence *coords;
    GEOSCoordSequence *coordsHoles;
    GEOSGeometry *lr;

    for (unsigned int i = 0; i < nShapes; i++)
    {
        shapeHandler->getShape(i, shapeObj);
        shapeParts = shapeObj.getParts();

        if (shapeObj.getType() != SHPT_POLYGON)
        {
            continue;
        }

        for (unsigned int partIndex = 0; partIndex < shapeParts.size(); partIndex++)
        {

            int nHoles = 0;
            xVertex.clear();
            yVertex.clear();
            xVertexHoles.clear();
            yVertexHoles.clear();

            std::vector<unsigned int> holesParts = shapeHandler->getHoles(i,partIndex);
            int offset = shapeObj.getPart(partIndex).offset;
            int length = shapeObj.getPart(partIndex).length;
            if (!shapeParts[partIndex].hole)
            {
                for (unsigned long v = 0; v < length; v++)
                {
                    xVertex.push_back(shapeObj.getVertex(v+offset).x);
                    yVertex.push_back(shapeObj.getVertex(v+offset).y);
                }
                if ( xVertex[0] != xVertex[xVertex.size()-1] )
                {
                    // Ring not closed add missing vertex
                    xVertex.push_back(xVertex[0]);
                    yVertex.push_back(yVertex[0]);
                }
                for (int holesIndex = 0; holesIndex < holesParts.size(); holesIndex++)
                {
                    int offset = shapeObj.getPart(holesParts[holesIndex]).offset;
                    int length = shapeObj.getPart(holesParts[holesIndex]).length;
                    std::vector<double> x;
                    std::vector<double> y;
                    for (unsigned long v = 0; v < length; v++)
                    {
                        x.push_back(shapeObj.getVertex(v+offset).x);
                        y.push_back(shapeObj.getVertex(v+offset).y);
                    }
                    if ( x[0] != x[x.size()-1] )
                    {
                        // Ring not closed add missing vertex
                        x.push_back(x[0]);
                        y.push_back(y[0]);
                    }
                    xVertexHoles.push_back(x);
                    yVertexHoles.push_back(y);
                    nHoles = nHoles + 1;
                }
                coords = GEOSCoordSeq_create(xVertex.size(),2);
                for (int j=0; j<xVertex.size(); j++)
                {
                    GEOSCoordSeq_setX(coords,j,xVertex[j]);
                    GEOSCoordSeq_setY(coords,j,yVertex[j]);
                }
                lr = GEOSGeom_createLinearRing(coords);
                for (int holeIndex = 0; holeIndex < nHoles; holeIndex++)
                {
                    coordsHoles = GEOSCoordSeq_create(xVertexHoles[holeIndex].size(),2);
                    for (int j=0; j<xVertexHoles[holeIndex].size(); j++)
                    {
                        GEOSCoordSeq_setX(coordsHoles,j,xVertexHoles[holeIndex][j]);
                        GEOSCoordSeq_setY(coordsHoles,j,yVertexHoles[holeIndex][j]);
                    }
                    holes.append(GEOSGeom_createLinearRing(coordsHoles));
                }
                if (lr != nullptr)
                {
                    // create Polygon from LinearRing
                    geometries.append(GEOSGeom_createPolygon(lr,holes.data(),nHoles));
                }
                else
                {
                    qDebug() << "lr is nullptr, i = " << i;
                }
            }
            else
            {
                continue;
            }

        }
        shapeParts.clear();
    }

    GEOSGeometry *collection = nullptr;
    if ( !geometries.isEmpty() )
    {
        if ( geometries.count() > 1 )
        {
            collection = GEOSGeom_createCollection(GEOS_MULTIPOLYGON, geometries.data(), geometries.count());
            //collection = GEOSGeom_createCollection(GEOS_GEOMETRYCOLLECTION, geometries.data(), geometries.count());
        }
        else
        {
            collection = geometries[0];
        }
   }
   if (collection == nullptr)
   {
        return nullptr;
   }
   return collection;

}


/*
GEOSGeometry *SHPObject_to_LineString(SHPObject *object)
{
    // Create a Coordinate sequence with object->nVertices coordinates of 2 dimensions.
    GEOSCoordSequence *coords = GEOSCoordSeq_create(object->nVertices,2);
    int i;

    assert(object->nParts == 1);
    for (i=0; i<object->nVertices; i++)
    {
        GEOSCoordSeq_setX(coords,i,object->padfX[i]);
        GEOSCoordSeq_setY(coords,i,object->padfY[i]);
    }
    return GEOSGeom_createLineString(coords);
}
*/

/*
GEOSGeometry * SHPObject_to_GeosPolygon_NoHoles(SHPObject *object)
{
    GEOSGeometry *lr;
    // Create a Coordinate sequence with object->nVertices coordinates of 2 dimensions.
    GEOSCoordSequence *coords = GEOSCoordSeq_create(object->nVertices,2);

    for (int i=0; i<object->nVertices; i++)
    {
        GEOSCoordSeq_setX(coords,i,object->padfX[i]);
        GEOSCoordSeq_setY(coords,i,object->padfY[i]);
    }
    // create LinearRing
    lr = GEOSGeom_createLinearRing(coords);
    // create Polygon from LinearRing (assuming no holes)
    return GEOSGeom_createPolygon(lr,nullptr,0);
}
*/

/*
GEOSGeometry *load_shapefile_as_collection(char *pathname)
{
    SHPHandle shape;
    int type, nobjs, i;
    double minBounds[4], maxBounds[4];
    GEOSGeometry **geometries;
    GEOSGeometry *collection;

    shape = SHPOpen(pathname,"rb");

    SHPGetInfo(shape,&nobjs,&type,minBounds,maxBounds);
    assert((type % 10) == SHPT_ARC);

    assert(geometries = (GEOSGeometry **) malloc(nobjs*sizeof(GEOSGeometry *)));

    for (i=0; i<nobjs ;i++)
    {
        SHPObject *object = SHPReadObject(shape,i);
        geometries[i] = SHPObject_to_GeosPolygon_NoHoles(object);
    }

    SHPClose(shape);

    collection = GEOSGeom_createCollection(GEOS_MULTIPOLYGON, geometries, nobjs);

    return collection;
}
*/


// OLD problem MultiPolygon
/*
GEOSGeometry * loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler)
{

    // Init GEOS
    GEOSMessageHandler error_function = nullptr, notice_function = nullptr;
    initGEOS(notice_function, error_function);

    GEOSGeometry **geometries;
    ShapeObject shapeObj;

    int nShapes = shapeHandler->getShapeCount();
    std::vector< std::vector<ShapeObject::Part>> shapeParts;
    geometries = (GEOSGeometry **) malloc(nShapes*sizeof(GEOSGeometry *));

    std::vector<double> xVertex;
    std::vector<double> yVertex;
    std::vector<std::vector <double> > xVertexHoles;
    std::vector<std::vector <double> > yVertexHoles;

    GEOSCoordSequence *coords;
    GEOSCoordSequence *coordsHoles;
    GEOSGeometry *lr;
    GEOSGeometry **holes = nullptr;

    for (unsigned int i = 0; i < nShapes; i++)
    {
        shapeHandler->getShape(i, shapeObj);
        shapeParts.push_back(shapeObj.getParts());
        int nHoles = 0;
        xVertex.clear();
        yVertex.clear();
        xVertexHoles.clear();
        yVertexHoles.clear();

        for (unsigned int partIndex = 0; partIndex < shapeParts[i].size(); partIndex++)
        {
            //qDebug() << "shapeParts[i].size() " << shapeParts[i].size();
            int offset = shapeObj.getPart(partIndex).offset;
            int length = shapeObj.getPart(partIndex).length;
            if (shapeParts[i][partIndex].hole)
            {

                std::vector<double> x;
                std::vector<double> y;
                for (unsigned long v = 0; v < length; v++)
                {
                    x.push_back(shapeObj.getVertex(v+offset).x);
                    y.push_back(shapeObj.getVertex(v+offset).y);
                }
                xVertexHoles.push_back(x);
                yVertexHoles.push_back(y);
                nHoles = nHoles + 1;

            }
            else
            {
                for (unsigned long v = 0; v < length; v++)
                {
                    xVertex.push_back(shapeObj.getVertex(v+offset).x);
                    yVertex.push_back(shapeObj.getVertex(v+offset).y);
                }
                if ( xVertex[offset] != xVertex[offset+length-1] )
                {
                // Ring not closed add missing vertex
                 xVertex.push_back(xVertex[offset]);
                 yVertex.push_back(yVertex[offset]);
               }
            }
        }
        if (nHoles == 0)
        {
            holes = nullptr;
        }
        else
        {
            holes = (GEOSGeometry **) malloc(nHoles * sizeof(GEOSGeometry *));
        }

        coords = GEOSCoordSeq_create(xVertex.size(),2);
        for (int j=0; j<xVertex.size(); j++)
        {
            GEOSCoordSeq_setX(coords,j,xVertex[j]);
            GEOSCoordSeq_setY(coords,j,yVertex[j]);
        }
        lr = GEOSGeom_createLinearRing(coords);

        for (int holeIndex = 0; holeIndex < nHoles; holeIndex++)
        {
            coordsHoles = GEOSCoordSeq_create(xVertexHoles[holeIndex].size(),2);
            for (int j=0; j<xVertexHoles[holeIndex].size(); j++)
            {
                GEOSCoordSeq_setX(coordsHoles,j,xVertexHoles[holeIndex][j]);
                GEOSCoordSeq_setY(coordsHoles,j,yVertexHoles[holeIndex][j]);
            }
            holes[holeIndex] = GEOSGeom_createLinearRing(coordsHoles);
        }
        if (lr != nullptr)
        {
            // create Polygon from LinearRing
            geometries[i] = GEOSGeom_createPolygon(lr,holes,nHoles);
            if (geometries[i] == nullptr)
            {
                qDebug() << "geometries[i] is nullptr, i = " << i;
            }
        }
        else
        {
            qDebug() << "lr is nullptr, i = " << i;
        }

    }
    GEOSGeometry *collection = GEOSGeom_createCollection(GEOS_MULTIPOLYGON, geometries, nShapes);
    if (collection == nullptr)
    {
        qDebug() << "collection is nullptr";
    }
    delete [] geometries;
    delete [] holes;
    return collection;
}
*/

/*
 * // Simple numeric test
GEOSGeometry * testIntersection()
{
// Init GEOS
    GEOSMessageHandler error_function = nullptr, notice_function = nullptr;
    initGEOS(notice_function, error_function);

    GEOSCoordSeq coordseq = nullptr, coordseqSecond = nullptr, coordseqIntersection = nullptr;
    GEOSGeom area_1 = nullptr, area_2 = nullptr, intersection = nullptr;
    GEOSGeometry *pol1;
    GEOSGeometry *pol2;

    coordseq = (GEOSCoordSeq) GEOSCoordSeq_create(5, 2);   //5 pointsbi-dimensional

    GEOSCoordSeq_setX(coordseq, 0, 42.46);    //upper left
    GEOSCoordSeq_setY(coordseq, 0, 131.80);
    GEOSCoordSeq_setX(coordseq, 1, 42.46);    //upper right
    GEOSCoordSeq_setY(coordseq, 1, 112.91);
    GEOSCoordSeq_setX(coordseq, 2, 21.96);    //lower right
    GEOSCoordSeq_setY(coordseq, 2, 112.91);
    GEOSCoordSeq_setX(coordseq, 3, 21.96);    //lower left
    GEOSCoordSeq_setY(coordseq, 3, 131.80);
    GEOSCoordSeq_setX(coordseq, 4, 42.46 );    //upper left
    GEOSCoordSeq_setY(coordseq, 4, 131.80);

    area_1 = GEOSGeom_createLinearRing(coordseq);

    pol1 = GEOSGeom_createPolygon(area_1, nullptr, 0);

    if((GEOSisEmpty(area_1) != 0) || (GEOSisValid(area_1) != 1)) {
        printf("No valid intersection found.\n");
        exit(2);    //invalid input parameter
    }

    coordseqSecond = (GEOSCoordSeq) GEOSCoordSeq_create(5, 2);   //5 pointsbi-dimensional

    GEOSCoordSeq_setX(coordseqSecond, 0, 43.22);    //upper left
    GEOSCoordSeq_setY(coordseqSecond, 0, 125.52);
    GEOSCoordSeq_setX(coordseqSecond, 1, 43.22);    //upper right
    GEOSCoordSeq_setY(coordseqSecond, 1, 106.47);
    GEOSCoordSeq_setX(coordseqSecond, 2, 22.71);    //lower right
    GEOSCoordSeq_setY(coordseqSecond, 2, 106.47);
    GEOSCoordSeq_setX(coordseqSecond, 3, 22.71);    //lower left
    GEOSCoordSeq_setY(coordseqSecond, 3, 125.52);
    GEOSCoordSeq_setX(coordseqSecond, 4, 43.22);    //upper left
    GEOSCoordSeq_setY(coordseqSecond, 4, 125.52);

    area_2 = GEOSGeom_createLinearRing(coordseqSecond);

    pol2 = GEOSGeom_createPolygon(area_2, nullptr, 0);

    if((GEOSisEmpty(area_2) != 0) || (GEOSisValid(area_2) != 1)) {
        printf("No valid intersection found.\n");
        exit(2);    //invalid input parameter
    }


    intersection = GEOSIntersection(pol1, pol2);

    if((GEOSisEmpty(intersection) != 0) || (GEOSisValid(intersection) !=1)) {
        printf("No valid intersection found.\n");
        exit(2);    //invalid input parameter
    }

    //Getting coords for the vertex
    unsigned int num;
    double xPoints[4];
    double yPoints[4];

    GEOSGeom geom;

    num = GEOSGetNumGeometries(intersection);
    printf("Geometries: %d\n",num);

    //GEOSCoordSeq_destroy(coordseq);
    coordseqIntersection = (GEOSCoordSeq) GEOSCoordSeq_create(2, 2);   //2 pointsbi-dimensional

    for(int i=0; i < num; i++) {
        geom = (GEOSGeom) GEOSGetGeometryN(intersection, i);

        coordseqIntersection = (GEOSCoordSeq) GEOSGeom_getCoordSeq(geom);

        GEOSCoordSeq_getX(coordseqIntersection, 0, &xPoints[i]);
        GEOSCoordSeq_getY(coordseqIntersection, 0, &yPoints[i]);
    }

    // Finalizzo GEOS
    finishGEOS();
}
*/


bool shapeToRaster(QString shapeFileName, std::string shapeField, QString resolution, QString proj, QString outputName, QString &errorStr)
{

    int error = -1;
    GDALAllRegister();
    QFileInfo file(outputName);
    QString ext = file.completeSuffix();

    std::string formatOption;
    if (mapExtensionShortName.contains(ext))
    {  
        formatOption = mapExtensionShortName.value(ext).toStdString();
    }
    else
    {
        errorStr = "Unknown output format";
        return false;
    }

    GDALDataset* shpDS;
    GDALDatasetH rasterizeDS;
    shpDS = (GDALDataset*)GDALOpenEx(shapeFileName.toStdString().data(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if( shpDS == nullptr )
    {
        errorStr = "Open shapefile failed";
        return false;
    }

    // projection
    QString noProj;
    char *pszProjection = nullptr;
    OGRSpatialReference srs;
    OGRSpatialReference * pOrigSrs = shpDS->GetLayer(0)->GetSpatialRef();
    if ( pOrigSrs )
    {
        srs = *pOrigSrs;
    }
    if ( srs.IsProjected() )
    {
        srs.exportToWkt( &pszProjection );
    }
    else
    {
        noProj = "EPSG:4326";
        pszProjection = strdup(noProj.toStdString().c_str());
        //GDALClose(shpDS);
        //errorStr = "Missing projection";
        //return false;
    }
    std::string outputNoReprojStd;

    if (proj.isEmpty())
    {
        outputNoReprojStd = outputName.toStdString();   // there is no reprojection to do
    }
    else
    {
        QString fileName = file.absolutePath() + "/" + file.baseName() + "_noreproj." + ext;
        outputNoReprojStd = fileName.toStdString();
    }

    std::string res = resolution.toStdString();

    // set options
    char *options[] = {strdup("-at"), strdup("-of"), strdup(formatOption.c_str()), strdup("-a"), strdup(shapeField.c_str()), strdup("-a_nodata"), strdup("-9999"),
                       strdup("-a_srs"), pszProjection, strdup("-tr"), strdup(res.c_str()), strdup(res.c_str()), strdup("-co"), strdup("COMPRESS=LZW"), nullptr};

    GDALRasterizeOptions *psOptions = GDALRasterizeOptionsNew(options, nullptr);
    if( psOptions == nullptr )
    {
        GDALClose(shpDS);
        errorStr = "psOptions is null";
        return false;
    }

    rasterizeDS = GDALRasterize(strdup(outputNoReprojStd.c_str()),nullptr,shpDS,psOptions,&error);

    if (rasterizeDS == nullptr || error == 1)
    {
        GDALClose(shpDS);
        GDALRasterizeOptionsFree(psOptions);
        CPLFree( pszProjection );
        return false;
    }

    // reprojection
    if (!proj.isEmpty())
    {
        GDALDatasetH hDstDS;
        GDALDriverH hDriver;
        GDALDataType eDT;
        // Create output with same datatype as first input band.
        eDT = GDALGetRasterDataType(GDALGetRasterBand(rasterizeDS,1));

        // Get output driver (GeoTIFF format)
        hDriver = GDALGetDriverByName( "GTiff" );
        if (hDriver == NULL)
        {
            errorStr = "Error GDALGetDriverByName";
            GDALClose(shpDS);
            GDALClose(rasterizeDS);
            GDALRasterizeOptionsFree(psOptions);
            CPLFree( pszProjection );
            return false;
        }

        // Get Source coordinate system.
        char *pszDstWKT = nullptr;
        pszDstWKT = strdup(proj.toStdString().c_str());
        // Create a transformer that maps from source pixel/line coordinates
        // to destination georeferenced coordinates (not destination
        // pixel line).  We do that by omitting the destination dataset
        // handle (setting it to NULL).
        void *hTransformArg;
        hTransformArg =
            GDALCreateGenImgProjTransformer( rasterizeDS, pszProjection, NULL, pszDstWKT,
                                             FALSE, 0, 1 );
        if ( hTransformArg == NULL )
        {
            errorStr = "Error GDALCreateGenImgProjTransformer";
            GDALClose(shpDS);
            GDALClose(rasterizeDS);
            GDALRasterizeOptionsFree(psOptions);
            CPLFree( pszProjection );
            return false;
        }

        // Get approximate output georeferenced bounds and resolution for file.
        double adfDstGeoTransform[6];
        int nPixels=0, nLines=0;
        CPLErr eErr;
        eErr = GDALSuggestedWarpOutput( rasterizeDS,
                                        GDALGenImgProjTransform, hTransformArg,
                                        adfDstGeoTransform, &nPixels, &nLines );
        if( eErr != CE_None )
        {
            errorStr = "Error GDALSuggestedWarpOutput";
            GDALClose(shpDS);
            GDALClose(rasterizeDS);
            GDALRasterizeOptionsFree(psOptions);
            CPLFree( pszProjection );
            return false;
        }
        GDALDestroyGenImgProjTransformer( hTransformArg );

        // Create the output file.
        hDstDS = GDALCreate( hDriver, strdup(outputName.toStdString().c_str()) , nPixels, nLines,
                             GDALGetRasterCount(rasterizeDS), eDT, NULL );
        if( hDstDS == NULL )
        {
            errorStr = "Error GDALCreate output reprojected";
            GDALClose(shpDS);
            GDALClose(rasterizeDS);
            GDALRasterizeOptionsFree(psOptions);
            CPLFree( pszProjection );
            return false;
        }

        // Write out the projection definition.
        GDALSetProjection( hDstDS, pszDstWKT );
        GDALSetGeoTransform( hDstDS, adfDstGeoTransform );

        // Initialize and execute the warp operation.
        eErr = GDALReprojectImage(rasterizeDS, pszProjection,
                                  hDstDS, pszDstWKT,
                                  GRA_Bilinear,
                                  0.0, 0.0,
                                  GDALTermProgress, NULL,
                                  NULL);
        if (eErr != CE_None)
        {
            errorStr =  CPLGetLastErrorMsg();
            GDALClose(shpDS);
            GDALClose(rasterizeDS);
            GDALRasterizeOptionsFree(psOptions);
            CPLFree( pszProjection );
            return false;
        }
        GDALClose( hDstDS );
    }
    GDALClose(shpDS);
    GDALClose(rasterizeDS);
    GDALRasterizeOptionsFree(psOptions);
    CPLFree( pszProjection );
    return true;
}

