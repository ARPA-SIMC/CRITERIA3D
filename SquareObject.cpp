#include "SquareObject.h"

#include <QKeyEvent>
#include <QStaticText>

#define NODATA -9999
#define RADIUS_WIDTH 3


SquareObject::SquareObject(qreal side, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    MapGraphicsObject(sizeIsZoomInvariant,parent), _fillColor(fillColor)
{
    _side = qMax<qreal>(side, 0.01);
    _currentValue = NODATA;
    _isText = false;

    this->setFlag(MapGraphicsObject::ObjectIsSelectable);
    this->setFlag(MapGraphicsObject::ObjectIsMovable);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable);
}

SquareObject::~SquareObject()
{
}

QRectF SquareObject::boundingRect() const
{
    return QRectF(-0.5*_side,
                  -0.5*_side,
                  _side,
                  _side);
}

void SquareObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setRenderHint(QPainter::Antialiasing,true);

    if (_isText)
    {
        if (_currentValue != NODATA)
        {
            QStaticText staticText = QStaticText( QString::number(_currentValue, 'f', 1) );
            staticText.setTextWidth(_side * RADIUS_WIDTH * 2);

            QTextOption myOption;
            myOption.setAlignment(Qt::AlignCenter);
            staticText.setTextOption(myOption);

            if (painter->font().bold())
            {
                QFont currentFont = painter->font();
                currentFont.setBold(false);
                painter->setFont(currentFont);
            }

            painter->scale(1,-1);
            painter->drawStaticText(-_side * RADIUS_WIDTH, -_side * 2, staticText);
        }
    }
    else
    {
        painter->setBrush(_fillColor);
        painter->drawRect(boundingRect());
    }
}

void SquareObject::setSide(qreal side)
{
    _side = side;
    emit this->redrawRequested();
}

void SquareObject::setFillColor(const QColor &color)
{
    if (_fillColor == color)
        return;

    _fillColor = color;
    emit this->redrawRequested();
}


void SquareObject::setCurrentValue(qreal currentValue)
{
    if (_currentValue == currentValue)
        return;

    _currentValue = currentValue;
    emit this->redrawRequested();
}


void SquareObject::showText(bool isShowText)
{
    if (_isText == isShowText)
        return;

    _isText = isShowText;
    emit this->redrawRequested();
}


//virtual from MapGraphicsObject
void SquareObject::keyReleaseEvent(QKeyEvent *event)
{
    if (event->matches(QKeySequence::Delete))
    {
        this->deleteLater();
        event->accept();
    }
    else
        event->ignore();
}


