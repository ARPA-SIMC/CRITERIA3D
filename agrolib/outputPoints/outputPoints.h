#ifndef OUTPUTPOINTS_H
#define OUTPUTPOINTS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <QString>

    class  Crit3DOutputPoint : public gis::Crit3DPoint
    {
    public:
        Crit3DOutputPoint();

        std::string id;
        double latitude;
        double longitude;

        bool active;
        bool selected;

        float currentValue;

        void initialize(const std::string& _id, bool isActive, double _latitude, double _longitude,
                        double _z, int zoneNumber);
    };


    bool importOutputPointsCsv(QString csvFileName, QList<QList<QString>> &data, QString &errorString);

    bool loadOutputPointListCsv(QString csvFileName, std::vector<Crit3DOutputPoint> &outputPointList,
                             int utmZone, QString &errorString);

    bool writeOutputPointListCsv(QString csvFileName, std::vector<Crit3DOutputPoint> &outputPointList, QString &errorString);


#endif // OUTPUTPOINTS_H
