#include "commonConstants.h"
#include "squareMarker.h"


SquareMarker::SquareMarker(qreal side,bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    SquareObject(side, sizeIsZoomInvariant, fillColor, parent)
{
    this->setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    this->setFlag(MapGraphicsObject::ObjectIsMovable, false);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable, false);

    _id = "";
    _currentValue = NODATA;
    _active = true;
}

void SquareMarker::setId(std::string id)
{
    _id = id;
}

std::string SquareMarker::id() const
{
    return _id;
}

void SquareMarker::setCurrentValue(float currentValue)
{
    _currentValue = currentValue;
}

bool SquareMarker::active() const
{
    return _active;
}

void SquareMarker::setActive(bool active)
{
    _active = active;
}

void SquareMarker::setToolTip()
{
    QString idpoint = QString::fromStdString(_id);

    QString toolTipText = QString("Point: <b> %1 </b> <br/>")
                            .arg(idpoint);

    if (_currentValue != NODATA)
    {
        QString value = QString::number(_currentValue);
        toolTipText += QString("value: <b> %1 </b>").arg(value);
    }

    SquareObject::setToolTip(toolTipText);
}


void SquareMarker::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{  
    Q_UNUSED(event)
    // TODO
}

void SquareMarker::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    Q_UNUSED(event)
    // TODO
}

