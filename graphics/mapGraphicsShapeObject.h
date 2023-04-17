#ifndef MAPGRAPHICSSHAPEOBJECT_H
#define MAPGRAPHICSSHAPEOBJECT_H

    #include "MapGraphicsObject.h"
    #include "MapGraphicsView.h"

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef MAP_H
        #include "geoMap.h"
    #endif

    struct LatLonPoint
    {
        double lon;
        double lat;
    };

    struct GeoBounds
    {
        LatLonPoint v0;
        LatLonPoint v1;
    };

    class MapGraphicsShapeObject : public MapGraphicsObject
    {
        Q_OBJECT

    private:
        MapGraphicsView* view;
        Crit3DShapeHandler* shapePointer;
        gis::Crit3DGeoMap* geoMap;
        QPointF referencePixel;

        unsigned int nrShapes;
        std::vector< std::vector<ShapeObject::Part>> shapeParts;
        std::vector< std::vector<GeoBounds>> geoBounds;
        std::vector< std::vector<LatLonPoint>> geoPoints;

        std::vector<float> values;
        std::vector<std::string> categories;

        bool isDrawing;
        bool isFill;

        void setMapExtents();
        void drawShape(QPainter* myPainter);
        void setPolygon(unsigned int i, unsigned int j, QPolygonF* polygon);
        int getCategoryIndex(std::string strValue);

    protected:
        /*!
         * \brief paint pure-virtual from MapGraphicsObject
         * \param painter a QPainter pointer
         * \param option a QStyleOptionGraphicsItem pointer
         * \param widget a QWidget pointer
         */
        void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

        /*!
         * \brief boundingRect pure-virtual from MapGraphicsObject
         * Defines the outer bounds of the item as a rectangle; all painting must be restricted to inside an item's bounding rect.
         * \return the bounding rect QRectF
         */
        QRectF boundingRect() const;


    public:
        /*!
         * \brief mapShapeObject constructor
         * \param view a MapGraphicsView pointer
         * \param parent MapGraphicsObject
         */
        explicit MapGraphicsShapeObject(MapGraphicsView* view, MapGraphicsObject *parent = nullptr);

        Crit3DColorScale* colorScale;

        void setDrawing(bool value);
        void updateCenter();
        void clear();

        bool initializeUTM(Crit3DShapeHandler* shapePtr);
        void setNumericValues(std::string fieldName);
        void setCategories(std::string fieldName);

        void setFill(bool value);
        Crit3DShapeHandler* getShapePointer();

        QPointF getPixel(const LatLonPoint &geoPoint);
    };


#endif // MAPGRAPHICSSHAPEOBJECT_H
