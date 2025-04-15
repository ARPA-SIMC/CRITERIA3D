#ifndef WELL_H
#define WELL_H

#include <QString>
#include <QList>
#include <QDate>
#include <QMap>

class Well
{
public:
    QMap<QDate, float> depths;

    Well();

    QString getId() const { return _id; }
    void setId(const QString &newId) { _id = newId; }

    double getUtmX() const { return _utmX; }
    void setUtmX(double newUtmX) { _utmX = newUtmX; }

    double getUtmY() const { return _utmY; }
    void setUtmY(double newUtmY) { _utmY = newUtmY; }

    double getLatitude() const { return _lat; }
    void setLatitude(double newLat) { _lat = newLat; }

    double getLongitude() const { return _lon; }
    void setLongitude(double newLon) { _lon = newLon; }

    void insertData(const QDate &myDate, float myValue) { depths.insert(myDate, myValue); }

    int getObsDepthNr() const { return depths.size(); }

    QDate getFirstObsDate();
    QDate getLastObsDate();

    int minValuesPerMonth();

private:
    QString _id;

    double _utmX, _utmY;
    double _lat, _lon;

    QDate _firstDate, _lastDate;
};


#endif // WELL_H
