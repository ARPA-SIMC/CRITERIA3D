#include "commonConstants.h"
#include "stationMarker.h"

#include <QMenu>

StationMarker::StationMarker(qreal radius,bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsView* view, MapGraphicsObject *parent) :
    CircleObject(radius, sizeIsZoomInvariant, fillColor, parent)
{
    this->setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    this->setFlag(MapGraphicsObject::ObjectIsMovable, false);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable, false);
    _view = view;
    _id = "";
    _name = "";
    _dataset = "";
    _altitude = NODATA;
    _municipality = "";
    _active = true;
}

void StationMarker::setId(std::string id)
{
    _id = id;
}

std::string StationMarker::id() const
{
    return _id;
}

void StationMarker::setName(const std::string &name)
{
    _name = name;
}

void StationMarker::setDataset(const std::string &dataset)
{
    _dataset = dataset;
}

void StationMarker::setAltitude(double altitude)
{
    _altitude = altitude;
}

void StationMarker::setMunicipality(const std::string &municipality)
{
    _municipality = municipality;
}

void StationMarker::setQuality(const quality::qualityType &quality)
{
    _quality = quality;
}

bool StationMarker::active() const
{
    return _active;
}

void StationMarker::setActive(bool active)
{
    _active = active;
}

void StationMarker::setToolTip()
{
    QString idpoint = QString::fromStdString(_id);
    QString name = QString::fromStdString(_name);
    QString dataset = QString::fromStdString(_dataset);
    QString altitude = QString::number(_altitude);
    QString municipality = QString::fromStdString(_municipality);

    QString toolTipText = QString("Point: <b> %1 </b> <br/> ID: %2 <br/> dataset: %3 <br/> altitude: %4 m <br/> municipality: %5")
                            .arg(name, idpoint, dataset, altitude, municipality);

    if (currentValue() != NODATA)
    {
        QString value = QString::number(currentValue());

        QString myQuality = "";
        if (_quality == quality::wrong_syntactic)
            myQuality = "WRONG DATA (syntax control)";
        if (_quality == quality::wrong_spatial)
            myQuality = "WRONG DATA (spatial control)";

        toolTipText = QString("value: <b> %1 <br/> %2 <br/> </b>").arg(value, myQuality) + toolTipText;
    }

    CircleObject::setToolTip(toolTipText);
}


void StationMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{  
    if (event->button() == Qt::RightButton)
    {
        bool isGrid = false;
        QMenu menu;
        QAction *openMeteoWidget = menu.addAction("Open new meteo widget");
        QAction *appendMeteoWidget = menu.addAction("Append to last meteo widget");

        QAction *selection =  menu.exec(QCursor::pos());

        if (selection != nullptr)
        {
            if (selection == openMeteoWidget)
            {
                emit newStationClicked(_id, _name, isGrid);
            }
            else if (selection == appendMeteoWidget)
            {
                emit appendStationClicked(_id, _name, isGrid);
            }
        }
    }
}

void StationMarker::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    Q_UNUSED(event)
}

