#ifndef GRIDCELLMARKER_H
#define GRIDCELLMARKER_H

    #include "MapGraphics_global.h"
    #include "PolygonObject.h"
    #include "MapGraphicsView.h"

    class Crit3DMeteoPoint;

    class GridCellMarker : public PolygonObject
    {
        Q_OBJECT

        public:
            explicit GridCellMarker(QPolygonF geoPoly, QColor fillColor, MapGraphicsObject *parent = nullptr);
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
            void newStationClicked(std::string);
            void appendStationClicked(std::string);
            void openCropClicked(std::string);

    };

#endif // GRIDCELLMARKER_H


