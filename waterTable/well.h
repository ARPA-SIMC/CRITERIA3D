#ifndef WELL_H
#define WELL_H

#include <QString>
#include <QList>
#include <QDate>
#include <QMap>

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

private:
    QString id;
    double utmX;
    double utmY;
    QMap<QDate, int> depths;
};

#endif // WELL_H
