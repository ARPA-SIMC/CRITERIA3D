#ifndef STATIONMARKER_H
#define STATIONMARKER_H

    #include "MapGraphics_global.h"
    #include "CircleObject.h"
    #include "MapGraphicsView.h"
    #include "quality.h"
    #include "meteo.h"

    class Crit3DMeteoPoint;

    enum callerSoftware{PRAGA_caller, CRITERIA3D_caller, other_caller};

    class StationMarker : public CircleObject
    {
        Q_OBJECT

        public:
            explicit StationMarker(qreal radius, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent = nullptr);

            void setToolTip();

            void setId(std::string id) { _id = id; }
            std::string id() const { return _id; }

            bool active() const { return _active; }
            void setActive(bool active) { _active = active; }

            void setName(const std::string &name) { _name = name; }

            void setDataset(const std::string &dataset) { _dataset = dataset; }

            void setAltitude(double altitude) { _altitude = altitude; }

            void setCallerSoftware(callerSoftware caller)
            { _caller = caller;}

            void setLapseRateCode(lapseRateCodeType code)
            { _lapseRateCode = code; }

            void setRegion(const std::string &region)
            { _region = region; }

            void setProvince(const std::string &province)
            { _province = province; }

            void setMunicipality(const std::string &municipality)
            { _municipality = municipality; }

            void setQuality(const quality::qualityType &quality)
            { _quality = quality; }

    private:
            std::string _id;
            std::string _name;
            std::string _dataset;
            std::string _region;
            std::string _province;
            std::string _municipality;

            double _altitude;
            float _currentValue;

            lapseRateCodeType _lapseRateCode;
            quality::qualityType _quality;
            callerSoftware _caller;

            bool _active;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:
            void newStationClicked(std::string, std::string, std::string, double, std::string, bool);
            void appendStationClicked(std::string, std::string, std::string, double, std::string, bool);
            void newPointStatisticsClicked(std::string, bool);
            void changeOrogCodeClicked(std::string, int);
            void newHomogeneityTestClicked(std::string);
            void newSynchronicityTestClicked(std::string);
            void setSynchronicityReferenceClicked(std::string);
            void markPoint(std::string);
            void unmarkPoint(std::string);

    };

#endif // STATIONMARKER_H


