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
            void setName(const std::string &name);

    private:
            MapGraphicsView* _view;
            std::string _id;
            std::string _name;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:
            void newStationClicked(std::string, bool);
            void appendStationClicked(std::string, bool);
            void openCropClicked(std::string);

    };

#endif // STATIONMARKER_H


