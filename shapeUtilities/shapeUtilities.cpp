#include "shapeUtilities.h"
#include <QFile>
#include <QFileInfo>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include <qdebug.h>

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

GEOSGeometry * loadShapeAsPolygon(Crit3DShapeHandler *shapeHandler)
{

    // Init GEOS
    GEOSMessageHandler error_function = nullptr, notice_function = nullptr;
    initGEOS(notice_function, error_function);

    GEOSGeometry **geometries;
    GEOSGeometry *collection;
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
    GEOSGeometry **holes;

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
        // create Polygon from LinearRing
        geometries[i] = GEOSGeom_createPolygon(lr,holes,nHoles);
        free(holes);
    }
    collection = GEOSGeom_createCollection(GEOS_GEOMETRYCOLLECTION, geometries, nShapes);
    return collection;
}
