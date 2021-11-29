#ifndef DIALOGNEWPOINT_H
#define DIALOGNEWPOINT_H

#include <QtWidgets>
#include <QString>
#include "gis.h"

class DialogNewPoint : public QDialog
{
     Q_OBJECT
public:
    DialogNewPoint(QList<QString> idList, gis::Crit3DRasterGrid DEM);
    ~DialogNewPoint();
    void done(bool res);
private:
    QList<QString> idList;
    gis::Crit3DRasterGrid DEM;
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
