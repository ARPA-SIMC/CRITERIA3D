#include "gdalShapeIntersection.h"
#include <QFileInfo>
#include <qdebug.h>
#include <ogrsf_frmts.h>
#include "ogr_spatialref.h"

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
/*
    QString error;
    OGRGeometry* firstPolygon = loadShapeAsPolygon(first, error);
    if (firstPolygon == nullptr || firstPolygon->IsEmpty())
    {
        qDebug() << "firstPolygon empty";
        return false;
    }
    if (!OGR_G_IsValid(firstPolygon))
    {
        OGR_G_MakeValid(firstPolygon);
        qDebug() << "firstPolygon not Valid";
    }
    OGRGeometry* secondPolygon = loadShapeAsPolygon(second, error);
    if (secondPolygon == nullptr || secondPolygon->IsEmpty())
    {
        qDebug() << "secondPolygon empty";
        return false;
    }
    if (!OGR_G_IsValid(secondPolygon))
    {
        OGR_G_MakeValid(secondPolygon);
        qDebug() << "secondPolygon not Valid";
    }
    OGRGeometry* ogrIntersection = firstPolygon->Intersection(secondPolygon);
    if (ogrIntersection == nullptr || ogrIntersection->IsEmpty())
    {
        OGRGeometryFactory::destroyGeometry(ogrIntersection);
        return false;
    }
*/

    GEOSGeometry* firstPolygon = loadShapeAsPolygon(first);
    if((GEOSisEmpty(firstPolygon)))
    {
        qDebug() << "first Polygon empty";
        return false;
    }

    if (GEOSisValid(firstPolygon) !=1)
    {
          qDebug() << "first Polygon is NOT Valid";
          qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(firstPolygon);
          firstPolygon = GEOSMakeValid(firstPolygon);
          qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(firstPolygon);
    }
   else
      qDebug() << "first Polygon is Valid";

    GEOSGeometry *secondPolygon = loadShapeAsPolygon(second);
    if((GEOSisEmpty(secondPolygon)))
    {
        qDebug() << "second Polygon empty";
        return false;
    }

    if (GEOSisValid(secondPolygon) !=1)
    {
          qDebug() << "second Polygon is NOT Valid";
          qDebug() << "Resulting geometry before is " << GEOSGeomToWKT(secondPolygon);
          secondPolygon = GEOSMakeValid(secondPolygon);
          qDebug() << "Resulting geometry after is " << GEOSGeomToWKT(secondPolygon);
    }
   else
      qDebug() << "second Polygon is Valid";

    GEOSContextHandle_t hGEOSCtxt = OGRGeometry::createGEOSContext();
    OGRGeometry* firstOGR = OGRGeometryFactory::createFromGEOS( hGEOSCtxt, firstPolygon );
    OGRGeometry* secondOGR = OGRGeometryFactory::createFromGEOS( hGEOSCtxt, secondPolygon );
    OGRGeometry* ogrIntersection = firstOGR->Intersection(secondOGR);
    if (ogrIntersection == nullptr || ogrIntersection->IsEmpty())
    {
        OGRGeometryFactory::destroyGeometry(ogrIntersection);
        return false;
    }

    //GEOSContextHandle_t hGEOSCtxt = OGRGeometry::createGEOSContext();
    *inteserctionGeom = ogrIntersection->exportToGEOS(hGEOSCtxt);

    //*inteserctionGeom = GEOSIntersection(firstPolygon, secondPolygon);
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
        //qDebug() << "Resulting geometry is " << GEOSGeomToWKT(*inteserctionGeom);
        return true;
    }

}

// OK controllata
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

            //interior rings
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

OGRGeometry* loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler, QString &errorStr)
{

    OGRDataSource *poDS = (OGRDataSource*)GDALOpenEx(shapeHandler->getFilepath().data(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if( poDS == nullptr )
    {
        errorStr = "Open shapefile failed";
        return nullptr;
    }
    OGRLayer *poLyr = poDS->GetLayer(0);
    if (poLyr == nullptr)
    {
        errorStr = "Failed to identify source layer from datasource.";
        OGRDataSource::DestroyDataSource(poDS);
        return nullptr;
    }

    OGRFeature *poFeat;
    OGRGeometry *poGeom = nullptr;
    while ((poFeat = poLyr->GetNextFeature()) != nullptr)
    {
        OGRGeometry* poSrcGeom = poFeat->GetGeometryRef();
        if (poSrcGeom)
        {
            OGRwkbGeometryType eType = wkbFlatten( poSrcGeom->getGeometryType() );

            if (poGeom == nullptr)
                poGeom = OGRGeometryFactory::createGeometry( wkbMultiPolygon );

            if( eType == wkbPolygon )
                ((OGRGeometryCollection*)poGeom)->addGeometry( poSrcGeom );
            else if( eType == wkbMultiPolygon )
            {
                int iGeom;
                int nGeomCount = OGR_G_GetGeometryCount( (OGRGeometryH)poSrcGeom );

                for( iGeom = 0; iGeom < nGeomCount; iGeom++ )
                {
                    ((OGRGeometryCollection*)poGeom)->addGeometry(
                                ((OGRGeometryCollection*)poSrcGeom)->getGeometryRef(iGeom) );
                }
            }
            else
            {
                errorStr = "ERROR: Geometry not of polygon type.\n";
                OGRGeometryFactory::destroyGeometry(poGeom);
                OGRFeature::DestroyFeature(poFeat);
                OGRDataSource::DestroyDataSource(poDS);
                return nullptr;
            }
        }

        OGRFeature::DestroyFeature(poFeat);
    }

    OGRDataSource::DestroyDataSource(poDS);

    return poGeom;

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
            //collection = GEOSGeom_createCollection(GEOS_MULTIPOLYGON, geometries.data(), geometries.count());
            collection = GEOSGeom_createCollection(GEOS_GEOMETRYCOLLECTION, geometries.data(), geometries.count());
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
