#ifndef STATIONMARKER_H
#define STATIONMARKER_H

    #include "MapGraphics_global.h"
    #include "CircleObject.h"
    #include "MapGraphicsView.h"

    class Crit3DMeteoPoint;

    class StationMarker : public CircleObject
    {
        Q_OBJECT

        public:
            explicit StationMarker(qreal radius, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsView* view, MapGraphicsObject *parent = nullptr);
            void setId(std::string id);
            void setToolTip(Crit3DMeteoPoint* meteoPoint_);
            std::string id() const;

    private:
            MapGraphicsView* _view;
            std::string _id;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            //void mouseDoubleClickEvent(QGraphicsSceneMouseEvent * event);
        signals:
            void newStationClicked();
            void appendStationClicked();

    };

#endif // STATIONMARKER_H


