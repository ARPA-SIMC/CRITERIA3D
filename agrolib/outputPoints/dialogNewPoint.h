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
private:
    QList<QString> idList;
    gis::Crit3DRasterGrid DEM;
    QTextEdit id;
    QTextEdit utmx;
    QTextEdit utmy;
    QTextEdit lat;
    QTextEdit lon;
    QTextEdit height;
    QPushButton computeUTMButton;
    QPushButton getFromDEMButton;
};

#endif // DIALOGNEWPOINT_H
