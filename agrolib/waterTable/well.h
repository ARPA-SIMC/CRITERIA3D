#ifndef WELL_H
#define WELL_H

#include <QString>
#include <QList>
#include <QDate>
#include <QMap>
#include "meteoPoint.h"

class Well
{
public:
    Well();
    QString getId() const;
    void setId(const QString &newId);

    double getUtmX() const;
    void setUtmX(double newUtmX);

    double getUtmY() const;
    void setUtmY(double newUtmY);

    void insertData(QDate myDate, int myValue);

    QDate getFirstDate();
    QDate getLastDate();

    int getDepthNr();

    QMap<QDate, int> getDepths() const;

private:
    QString id;
    double utmX;
    double utmY;
    QMap<QDate, int> depths;
    QDate firstDate;
    QDate lastDate;
};

#endif // WELL_H
