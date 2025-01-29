#include "commonConstants.h"
#include "basicMath.h"
#include "stationMarker.h"
#include "qdebug.h"

#include <math.h>
#include <QMenu>

StationMarker::StationMarker(qreal radius,bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    CircleObject(radius, sizeIsZoomInvariant, fillColor, parent)
{
    this->setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    this->setFlag(MapGraphicsObject::ObjectIsMovable, false);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable, false);

    _id = "";
    _name = "";
    _dataset = "";
    _municipality = "";
    _altitude = NODATA;
    _lapseRateCode = primary;

    _caller = PRAGA_caller;
    _active = true;
}


void StationMarker::setToolTip()
{
    QString idpoint = QString::fromStdString(_id);
    QString name = QString::fromStdString(_name);
    QString dataset = QString::fromStdString(_dataset);
    QString altitude = QString::number(_altitude);
    QString region = QString::fromStdString(_region);
    QString province = QString::fromStdString(_province);
    QString municipality = QString::fromStdString(_municipality);
    QString lapseRateName = QString::fromStdString(getLapseRateCodeName(_lapseRateCode));

    QString toolTipText = QString("Point: <b> %1 </b> <br/> ID: %2 <br/> dataset: %3 <br/> altitude: %4 m <br/> region: %5 <br/> province: %6 <br/> <br/> municipality: %7 <br/> <br/> lapse rate code: %8")
                            .arg(name, idpoint, dataset, altitude, region, province, municipality, lapseRateName);

    double value = currentValue();
    if (! isEqual(value, NODATA) || isMarked())
    {
        QString valueStr;
        if (fabs(value) <= 1)
            valueStr = QString::number(value, 'f', 2);
        else
            valueStr = QString::number(value, 'f', 1);

        QString myQuality = "";
        if (_quality == quality::wrong_syntactic)
            myQuality = "WRONG DATA (syntax control)";
        if (_quality == quality::wrong_spatial)
            myQuality = "WRONG DATA (spatial control)";

        toolTipText = QString("value: <b> %1 <br/> %2 <br/> </b>").arg(valueStr, myQuality) + toolTipText;
    }

    CircleObject::setToolTip(toolTipText);
}


void StationMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{  
    if (event->button() == Qt::RightButton)
    {
        if (_caller == PRAGA_caller)
        {
            QMenu menu;
            QAction *openMeteoWidget = menu.addAction("Open a new meteo widget");
            QAction *appendMeteoWidget = menu.addAction("Append to the lastest meteo widget");
            menu.addSeparator();
            QAction *openPointStatisticsWidget = menu.addAction("Open point statistics widget");
            QAction *openHomogeneityWidget = menu.addAction("Open homogeneity test widget");
            menu.addSeparator();
            QAction *openSynchronicityWidget = menu.addAction("Open synchronicity test widget");
            QAction *setSynchronicityReferencePoint = menu.addAction("Set as synchronicity reference point");
            menu.addSeparator();
            QAction *actionMarkPoint = menu.addAction( "Mark point" );
            QAction *actionUnmarkPoint = menu.addAction( "Unmark point" );
            menu.addSeparator();
            QMenu *orogCodeSubMenu;
            orogCodeSubMenu = menu.addMenu("Orog code");
            QAction *actionOrogCode_primary = orogCodeSubMenu->addAction( "Set as primary station" );
            QAction *actionOrogCode_secondary = orogCodeSubMenu->addAction( "Set as secondary station" );
            QAction *actionOrogCode_supplemental = orogCodeSubMenu->addAction( "Set as supplemental station" );

            QAction *selection =  menu.exec(QCursor::pos());

            if (selection != nullptr)
            {
                bool isGrid = false;
                std::string lapseRateCode = getLapseRateCodeName(_lapseRateCode);
                if (selection == openMeteoWidget)
                {
                    emit newStationClicked(_id, _name, _dataset, _altitude, lapseRateCode, isGrid);
                }
                else if (selection == appendMeteoWidget)
                {
                    emit appendStationClicked(_id, _name, _dataset, _altitude, lapseRateCode, isGrid);
                }
                else if (selection == openPointStatisticsWidget)
                {
                    emit newPointStatisticsClicked(_id, isGrid);
                }
                else if (selection == openHomogeneityWidget)
                {
                    emit newHomogeneityTestClicked(_id);
                }
                else if (selection == openSynchronicityWidget)
                {
                    emit newSynchronicityTestClicked(_id);
                }
                else if (selection == setSynchronicityReferencePoint)
                {
                    emit setSynchronicityReferenceClicked(_id);
                }
                else if (selection == actionOrogCode_primary)
                {
                    emit changeOrogCodeClicked(_id, 0);
                }
                else if (selection == actionOrogCode_secondary)
                {
                    emit changeOrogCodeClicked(_id, 1);
                }
                else if (selection == actionOrogCode_supplemental)
                {
                    emit changeOrogCodeClicked(_id, 2);
                }
                else if (selection == actionMarkPoint)
                {
                    emit markPoint(_id);
                }
                else if (selection == actionUnmarkPoint)
                {
                    emit unmarkPoint(_id);
                }
            }
        }
        else
        {
            // Other Software
            QMenu menu;
            QAction *openMeteoWidget = menu.addAction("Open a new meteo widget");
            QAction *appendMeteoWidget = menu.addAction("Append to the lastest meteo widget");
            menu.addSeparator();

            QAction *selection =  menu.exec(QCursor::pos());

            if (selection != nullptr)
            {
                bool isGrid = false;
                std::string lapseRateCode = getLapseRateCodeName(_lapseRateCode);
                if (selection == openMeteoWidget)
                {
                    emit newStationClicked(_id, _name, _dataset, _altitude, lapseRateCode, isGrid);
                }
                else if (selection == appendMeteoWidget)
                {
                    emit appendStationClicked(_id, _name, _dataset, _altitude, lapseRateCode, isGrid);
                }
            }
        }
    }
}

void StationMarker::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    Q_UNUSED(event)
}

