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

    QString getId() const { return id; }
    void setId(const QString &newId) { id = newId; }

    double getUtmX() const { return utmX; }
    void setUtmX(double newUtmX) { utmX = newUtmX; }

    double getUtmY() const { return utmY; }
    void setUtmY(double newUtmY) { utmY = newUtmY; }

    double getLatitude() const { return lat; }
    void setLatitude(double newLat) { lat = newLat; }

    double getLongitude() const { return lon; }
    void setLongitude(double newLon) { lon = newLon; }

    void insertData(QDate myDate, float myValue)
    { depths.insert(myDate, myValue); }

    int getObsDepthNr() const { return depths.size(); }

    QDate getFirstDate();
    QDate getLastDate();

    int minValuesPerMonth();

private:
    QString id;

    double utmX, utmY;
    double lat, lon;

    QDate firstDate;
    QDate lastDate;
};

#endif // WELL_H
