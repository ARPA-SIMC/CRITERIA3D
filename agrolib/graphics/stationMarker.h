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
            void setToolTip();
            std::string id() const;
            void setName(const std::string &name);
            void setDataset(const std::string &dataset);
            void setAltitude(double altitude);
            void setLapseRateCode(lapseRateCodeType code);
            void setMunicipality(const std::string &municipality);
            void setQuality(const quality::qualityType &quality);
            bool active() const;
            void setActive(bool active);

    private:
            MapGraphicsView* _view;
            std::string _id;
            std::string _name;
            std::string _dataset;
            double _altitude;
            lapseRateCodeType _lapseRateCode;
            std::string _municipality;
            float _currentValue;
            quality::qualityType _quality;
            bool _active;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:
            void newStationClicked(std::string, std::string, bool);
            void appendStationClicked(std::string, std::string, bool);
            void newPointStatisticsClicked(std::string, bool);
            void changeOrogCodeClicked(std::string, int);
            void newHomogeneityTestClicked(std::string);
            void newSynchronicityTestClicked(std::string);
            void setSynchronicityReferenceClicked(std::string);

    };

#endif // STATIONMARKER_H


