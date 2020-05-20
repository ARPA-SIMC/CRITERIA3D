#ifndef STATIONMARKER_H
#define STATIONMARKER_H

    #include "MapGraphics_global.h"
    #include "CircleObject.h"
    #include "MapGraphicsView.h"
    #include "quality.h"

    class Crit3DMeteoPoint;

    class StationMarker : public CircleObject
    {
        Q_OBJECT

        public:
            explicit StationMarker(qreal radius, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsView* view, MapGraphicsObject *parent = nullptr);
            void setId(std::string id);
            //void setToolTip(Crit3DMeteoPoint* meteoPoint_);
            void setToolTip();
            //QString getToolTipText();
            std::string id() const;
            void setName(const std::string &name);
            void setDataset(const std::string &dataset);
            void setAltitude(double altitude);
            void setMunicipality(const std::string &municipality);
            void setQuality(const quality::qualityType &quality);
            void setCurrentValue(float currentValue);

    private:
            MapGraphicsView* _view;
            std::string _id;
            std::string _name;
            std::string _dataset;
            double _altitude;
            std::string _municipality;
            float _currentValue;
            quality::qualityType _quality;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:
            void newStationClicked(std::string, std::string, bool);
            void appendStationClicked(std::string, std::string, bool);
            void openCropClicked(std::string);

    };

#endif // STATIONMARKER_H


