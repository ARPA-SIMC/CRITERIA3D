#include "SquareObject.h"

#include <QKeyEvent>
#include <QStaticText>

#define NODATA -9999
#define TEXT_SIDE 2


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
    return QRectF(-_side * TEXT_SIDE,
                  -_side * 0.5,
                  _side * TEXT_SIDE * 2,
                  _side);
}



void SquareObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setRenderHint(QPainter::Antialiasing,true);

    if (_isText)
    {
        QString myStr = "";
        if (_isId)
        {
            myStr = _id;
        }
        else if (_currentValue != NODATA)
        {
            myStr =  QString::number(_currentValue, 'f', 1);
        }
        QStaticText staticText = QStaticText(myStr);
        staticText.setTextWidth(_side * TEXT_SIDE);

        QTextOption myOption;
        myOption.setAlignment(Qt::AlignCenter);
        staticText.setTextOption(myOption);

        if (! painter->font().bold())
        {
            QFont currentFont = painter->font();
            currentFont.setBold(true);
            painter->setFont(currentFont);
        }

        painter->scale(1,-1);
        painter->drawStaticText(-_side * TEXT_SIDE * 0.5, -_side, staticText);
    }
    else
    {
        painter->setBrush(_fillColor);
        painter->drawRect( QRectF(-_side * 0.5, -_side * 0.5, _side, _side) );
    }
}

void SquareObject::setId(const QString &id)
{
    if (_id == id)
        return;

    _id = id;
    emit this->redrawRequested();
}

void SquareObject::setSide(qreal side)
{
    if (_side == side)
        return;

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

void SquareObject::showId(bool isShowId)
{
    if (_isId == isShowId)
        return;

    _isId = isShowId;
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


