#ifndef DIALOGNEWPOINT_H
#define DIALOGNEWPOINT_H

#include <QtWidgets>
#include <QString>
#include "gis.h"

class DialogNewPoint : public QDialog
{
     Q_OBJECT
public:
    DialogNewPoint(const QList<QString>& _idList, const gis::Crit3DGisSettings& _gisSettings, gis::Crit3DRasterGrid* _DEMptr);
    ~DialogNewPoint();

    QString getId() { return id.text(); }
    double getLat() { return lat.text().toDouble(); }
    double getLon() { return lon.text().toDouble(); }
    double getHeight() { return height.text().toDouble(); }

    void setDEM();
    void computeUTM();
    void getFromDEM();

    void done(int res);

private:
    QList<QString> idList;

    gis::Crit3DGisSettings gisSettings;
    gis::Crit3DRasterGrid* DEMpointer;

    QLineEdit id, utmx, utmy;
    QLineEdit lat, lon, height;

    QPushButton computeUTMButton;
    QPushButton getFromDEMButton;
};

#endif // DIALOGNEWPOINT_H
