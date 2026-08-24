#include "mapGraphicsShapeObject.h"
#include "commonConstants.h"
#include "basicMath.h"


#define MAPBORDER 10


MapGraphicsShapeObject::MapGraphicsShapeObject(MapGraphicsView* _view, MapGraphicsObject *parent) :
    MapGraphicsObject(true, parent)
{
    setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    setFlag(MapGraphicsObject::ObjectIsMovable, false);
    setFlag(MapGraphicsObject::ObjectIsFocusable);
    view = _view;

    colorScale = new Crit3DColorScale();

    geoMap = new gis::Crit3DGeoMap();
    shapePointer = nullptr;

    _isDrawing = false;
    _isFill = false;
    _isSelectedRed = false;

    _nrShapes = 0;
    _selectedShape = NODATA;

    updateCenter();
}


/*!
\brief If sizeIsZoomInvariant() is true, this should return the size of the
 rectangle you want in PIXELS. If false, this should return the size of the rectangle in METERS. The
 rectangle should be centered at (0,0) regardless.
*/
QRectF MapGraphicsShapeObject::boundingRect() const
{
     return QRectF( -this->view->width() * 0.5, -this->view->height() * 0.5,
                     this->view->width() * 1.0,  this->view->height() * 1.0);
}


void MapGraphicsShapeObject::updateCenter()
{
    int widthPixels = view->width() - MAPBORDER*2;
    int heightPixels = view->height() - MAPBORDER*2;
    QPointF newCenter = view->mapToScene(QPoint(widthPixels/2, heightPixels/2));

    // reference point
    geoMap->referencePoint.latitude = newCenter.y();
    geoMap->referencePoint.longitude = newCenter.x();

    // reference pixel
    referencePixel = view->tileSource()->ll2qgs(newCenter, view->zoomLevel());

    if (_isDrawing)
        setPos(newCenter);
}


void MapGraphicsShapeObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    if (_isDrawing)
    {
        setMapExtents();

        if (shapePointer != nullptr)
        {
            drawShape(painter);
        }
    }
}


/*!
\brief convert a point in geo (lat,lon) coordinates
 in pixel (local object) coordinates
*/
QPointF MapGraphicsShapeObject::getPixel(const LatLonPoint &geoPoint)
{
    QPointF point = QPointF(geoPoint.lon, geoPoint.lat);
    QPointF pixel = this->view->tileSource()->ll2qgs(point, this->view->zoomLevel());
    pixel.setX(pixel.x() - this->referencePixel.x());
    pixel.setY(this->referencePixel.y() - pixel.y());
    return pixel;
}


void MapGraphicsShapeObject::setPolygon(unsigned int i,
                                        unsigned int j,
                                        QPolygonF* polygon)
{
    if (polygon == nullptr)
        return;

    polygon->clear();

    const unsigned long offset = shapeParts[i][j].offset;
    const unsigned long length = shapeParts[i][j].length;

    bool hasPreviousPoint = false;
    QPointF previousPoint;

    for (unsigned long v = 0; v < length; ++v)
    {
        const unsigned long vertexIndex = offset + v;
        const QPointF point = getPixel(geoPoints[i][vertexIndex]);

        if (!hasPreviousPoint || point != previousPoint)
        {
            polygon->append(point);
            previousPoint = point;
            hasPreviousPoint = true;
        }
    }
}


void MapGraphicsShapeObject::drawShape(QPainter* myPainter)
{
    if (! myPainter)
        return;

    static const QColor gray(64,64,64);
    static const QColor red(Qt::red);
    static const QColor black(Qt::black);

    for (unsigned long i = 0; i < _nrShapes; i++)
    {
        QPen myPen;
        if (i != _selectedShape)
        {
            myPen.setColor(gray);
            myPen.setWidth(1);
        }
        else
        {
            if (_isSelectedRed)
                myPen.setColor(red);
            else
                myPen.setColor(black);

            myPen.setWidth(2);
        }

        myPainter->setPen(myPen);
        myPainter->setBrush(Qt::NoBrush);

        if (_isFill && values[i] != NODATA)
        {
            Crit3DColor* myColor = colorScale->getColor(values[i]);
            QColor color(myColor->red, myColor->green, myColor->blue);
            myPainter->setBrush(color);

            if (i != _selectedShape)
            {
                myPainter->setPen(color);
            }
        }

        for (unsigned int j = 0; j < shapeParts[i].size(); j++)
        {
            if (shapeParts[i][j].hole)
                continue;

            if (geoBounds[i][j].v0.lon > geoMap->topRight.longitude
                    || geoBounds[i][j].v0.lat > geoMap->topRight.latitude
                    || geoBounds[i][j].v1.lon < geoMap->bottomLeft.longitude
                    || geoBounds[i][j].v1.lat < geoMap->bottomLeft.latitude)
                continue;

            QPolygonF polygon;
            setPolygon(i, j, &polygon);

            const std::vector<unsigned int>& myHoles = shapePointer->getHoles(i, j);

            if (myHoles.empty())
            {
                myPainter->drawPolygon(polygon);
            }
            else
            {
                QPainterPath path;
                path.setFillRule(Qt::OddEvenFill);

                path.addPolygon(polygon);

                for (unsigned int k = 0; k < myHoles.size(); ++k)
                {
                    setPolygon(i, myHoles[k], &polygon);
                    path.addPolygon(polygon);
                }

                myPainter->drawPath(path);
            }
        }
    }
}


bool MapGraphicsShapeObject::initializeUTM(Crit3DShapeHandler* shapePtr)
{
    if (shapePtr == nullptr)
        return false;

    const int zoneNumber = shapePtr->getUtmZone();
    if ((zoneNumber < 1) || (zoneNumber > 60))
        return false;

    clear();

    shapePointer = shapePtr;

    updateCenter();

    _nrShapes = unsigned(shapePointer->getShapeCount());
    shapeParts.resize(_nrShapes);
    geoBounds.resize(_nrShapes);
    geoPoints.resize(_nrShapes);
    values.resize(_nrShapes);

    const double refLatitude = geoMap->referencePoint.latitude;

    for (unsigned int i = 0; i < _nrShapes; i++)
    {
        ShapeObject myShape;
        double lat, lon;

        shapePointer->getShape(int(i), myShape);
        shapeParts[i] = myShape.getParts();

        // intialize values
        values[i] = NODATA;

        unsigned int nrParts = myShape.getPartCount();
        geoBounds[i].resize(nrParts);

        for (unsigned int j = 0; j < nrParts; j++)
        {
            // bounds
            Box<double>* bounds = &(shapeParts[i][j].boundsPart);
            gis::utmToLatLon(zoneNumber, refLatitude, bounds->xmin, bounds->ymin, &lat, &lon);
            geoBounds[i][j].v0.lat = lat;
            geoBounds[i][j].v0.lon = lon;

            gis::utmToLatLon(zoneNumber, refLatitude, bounds->xmax, bounds->ymax, &lat, &lon);
            geoBounds[i][j].v1.lat = lat;
            geoBounds[i][j].v1.lon = lon;
        }

        // vertices
        unsigned long nrVertices = myShape.getVertexCount();
        geoPoints[i].resize(nrVertices);

        const Point<double> *pointPtr = myShape.getVertices();
        for (unsigned long j = 0; j < nrVertices; j++)
        {
            gis::utmToLatLon(zoneNumber, refLatitude, pointPtr->x, pointPtr->y, &lat, &lon);
            geoPoints[i][j].lat = lat;
            geoPoints[i][j].lon = lon;
            pointPtr++;
        }
    }

    setDrawing(true);
    return true;
}


// warning: call after initializeUTM
void MapGraphicsShapeObject::setNumericValues(std::string fieldName)
{
    // set values
    float firstValue = NODATA;
    for (unsigned int i = 0; i < _nrShapes; i++)
    {
        values[i] = float(shapePointer->getNumericValue(signed(i), fieldName));

        if (isEqual(firstValue, NODATA) && (! isEqual(values[i], NODATA)))
            firstValue = values[i];
    }

    // set min/max
    colorScale->setRange(firstValue, firstValue);
    if (! isEqual(firstValue, NODATA))
    {
        for (unsigned int i = 0; i < _nrShapes; i++)
            if (! isEqual(values[i], NODATA))
            {
                colorScale->setMinimum(MINVALUE(colorScale->minimum(), values[i]));
                colorScale->setMaximum(MAXVALUE(colorScale->maximum(), values[i]));
            }
    }
}


int MapGraphicsShapeObject::getCategoryIndex(std::string strValue)
{
    for (unsigned int i = 0; i < categories.size(); i++)
    {
        if (categories[i] == strValue) return signed(i);
    }

    return NODATA;
}


// warning: call after initializeUTM
void MapGraphicsShapeObject::setCategories(std::string fieldName)
{
    // fill categories and set values(index of categories)
    categories.clear();
    for (unsigned int i = 0; i < _nrShapes; i++)
    {
        std::string strValue = shapePointer->getStringValue(signed(i), fieldName);

        if (strValue != "")
        {
            int index = getCategoryIndex(strValue);
            if (index != NODATA)
            {
                values[i] = index+1;
            }
            else
            {
                categories.push_back(strValue);
                values[i] = categories.size();
            }
        }
        else values[i] = NODATA;
    }

    // define min/max
    if (! categories.empty())
    {
        colorScale->setRange(1, float(categories.size()));
    }
    else
    {
        colorScale->setRange(NODATA, NODATA);
    }
}


void MapGraphicsShapeObject::setMapExtents()
{
    int widthPixels = view->width() - MAPBORDER*2;
    int heightPixels = view->height() - MAPBORDER*2;
    QPointF botLeft = view->mapToScene(QPoint(0, heightPixels));
    QPointF topRight = view->mapToScene(QPoint(widthPixels, 0));

    geoMap->bottomLeft.longitude = MAXVALUE(-180, botLeft.x());
    geoMap->bottomLeft.latitude = MAXVALUE(-84, botLeft.y());
    geoMap->topRight.longitude = MINVALUE(180, topRight.x());
    geoMap->topRight.latitude = MINVALUE(84, topRight.y());
}


void MapGraphicsShapeObject::clear()
{
    setDrawing(false);

    shapeParts.clear();
    geoBounds.clear();
    geoPoints.clear();
    values.clear();
    categories.clear();

    _nrShapes = 0;
    _selectedShape = NODATA;

    shapePointer = nullptr;

    colorScale->setRange(NODATA, NODATA);
}
