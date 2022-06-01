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

    void done(int res);

    void setDEM();
    void computeUTM();
    void getFromDEM();
    QString getId();
    double getLat();
    double getLon();
    double getHeight();

private:
    QList<QString> idList;
    gis::Crit3DGisSettings gisSettings;
    gis::Crit3DRasterGrid* DEMpointer;
    QLineEdit id;
    QLineEdit utmx;
    QLineEdit utmy;
    QLineEdit lat;
    QLineEdit lon;
    QLineEdit height;
    QPushButton computeUTMButton;
    QPushButton getFromDEMButton;
};

#endif // DIALOGNEWPOINT_H
