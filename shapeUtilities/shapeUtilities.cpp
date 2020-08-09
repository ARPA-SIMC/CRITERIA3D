#include "shapeUtilities.h"
#include <QFile>
#include <QFileInfo>
#include <qdebug.h>

#ifdef GDAL
    #include <gdal_priv.h>
    #include <ogrsf_frmts.h>
#endif


// make a copy of shapefile and return cloned shapefile path
QString cloneShapeFile(QString refFileName, QString newFileName)
{
    QFileInfo refFileInfo(refFileName);
    QFileInfo newFileInfo(newFileName);

    QString refFile = refFileInfo.absolutePath() + "/" + refFileInfo.baseName();
    QString newFile = newFileInfo.absolutePath() + "/" + newFileInfo.baseName();

    QFile::remove(newFile + ".dbf");
    QFile::copy(refFile +".dbf", newFile +".dbf");

    QFile::remove(newFile +".shp");
    QFile::copy(refFile +".shp", newFile +".shp");

    QFile::remove(newFile +".shx");
    QFile::copy(refFile +".shx", newFile +".shx");

    QFile::remove(newFile +".prj");
    QFile::copy(refFile +".prj", newFile +".prj");

    return(newFile + ".shp");
}


bool cleanShapeFile(Crit3DShapeHandler *shapeHandler)
{
    if (! shapeHandler->existRecordDeleted()) return true;

    QFileInfo fileInfo(QString::fromStdString(shapeHandler->getFilepath()));
    QString refFile = fileInfo.absolutePath() + "/" + fileInfo.baseName();
    QString tmpFile = refFile + "_temp";

    shapeHandler->packSHP(tmpFile.toStdString());
    shapeHandler->packDBF(tmpFile.toStdString());
    shapeHandler->close();

    QFile::remove(refFile + ".dbf");
    QFile::copy(tmpFile + ".dbf", refFile + ".dbf");
    QFile::remove(tmpFile + ".dbf");

    QFile::remove(refFile + ".shp");
    QFile::copy(tmpFile + ".shp", refFile + ".shp");
    QFile::remove(tmpFile + ".shp");

    QFile::remove(refFile + ".shx");
    QFile::copy(tmpFile + ".shx", refFile + ".shx");
    QFile::remove(tmpFile + ".shx");

    return shapeHandler->open(shapeHandler->getFilepath());
}


#ifdef GDAL
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
                if (lr != NULL)
                {
                    // create Polygon from LinearRing
                    geometries.append(GEOSGeom_createPolygon(lr,holes.data(),nHoles));
                }
                else
                {
                    qDebug() << "lr is NULL, i = " << i;
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
   if (collection == NULL)
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
    return GEOSGeom_createPolygon(lr,NULL,0);
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
            holes = NULL;
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
        if (lr != NULL)
        {
            // create Polygon from LinearRing
            geometries[i] = GEOSGeom_createPolygon(lr,holes,nHoles);
            if (geometries[i] == NULL)
            {
                qDebug() << "geometries[i] is NULL, i = " << i;
            }
        }
        else
        {
            qDebug() << "lr is NULL, i = " << i;
        }

    }
    GEOSGeometry *collection = GEOSGeom_createCollection(GEOS_MULTIPOLYGON, geometries, nShapes);
    if (collection == NULL)
    {
        qDebug() << "collection is NULL";
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

    GEOSCoordSeq coordseq = NULL, coordseqSecond = NULL, coordseqIntersection = NULL;
    GEOSGeom area_1 = NULL, area_2 = NULL, intersection = NULL;
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

    pol1 = GEOSGeom_createPolygon(area_1, NULL, 0);

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

    pol2 = GEOSGeom_createPolygon(area_2, NULL, 0);

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
#endif //GDAL
