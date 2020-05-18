#ifndef GRIDCELLMARKER_H
#define GRIDCELLMARKER_H

    #include "MapGraphics_global.h"
    #include "PolygonObject.h"
    #include "MapGraphicsView.h"
    #include "stationMarker.h"

    class Crit3DMeteoPoint;

    class GridCellMarker : public PolygonObject
    {
        Q_OBJECT

        public:
            explicit GridCellMarker(QPolygonF geoPoly, QColor fillColor, MapGraphicsView* view, MapGraphicsObject *parent = nullptr);
            void setId(std::string id);
            void setToolTip(Crit3DMeteoPoint* meteoPoint_);
            std::string id() const;
            void setName(const std::string &name);
            void setPointList(const QList<StationMarker *> &value);


    private:
            MapGraphicsView* _view;
            std::string _id;
            std::string _name;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

//            bool eventFilter(QObject *obj, QEvent *event);

        signals:
            void newCellClicked(std::string, std::string, bool);
            void appendCellClicked(std::string, std::string, bool);
            void openCropClicked(std::string);

    };

#endif // GRIDCELLMARKER_H


