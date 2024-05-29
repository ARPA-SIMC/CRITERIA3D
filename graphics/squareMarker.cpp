#include "commonConstants.h"
#include "squareMarker.h"


SquareMarker::SquareMarker(qreal side, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    SquareObject(side, sizeIsZoomInvariant, fillColor, parent)
{
    setFlag(MapGraphicsObject::ObjectIsSelectable, false);
    setFlag(MapGraphicsObject::ObjectIsMovable, false);
    setFlag(MapGraphicsObject::ObjectIsFocusable, false);

    _id = "";
    _active = true;
}


void SquareMarker::setToolTip()
{
    QString idpoint = QString::fromStdString(_id);

    QString toolTipText = QString("Point: <b> %1 </b> <br/>")
                            .arg(idpoint);

    if (currentValue() != NODATA)
    {
        QString value = QString::number(currentValue());
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

