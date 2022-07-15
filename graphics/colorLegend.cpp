#include "commonConstants.h"
#include "basicMath.h"
#include "colorLegend.h"
#include <QPainter>
#include <cmath>


ColorLegend::ColorLegend(QWidget *parent) :
    QWidget(parent)
{
    this->colorScale = nullptr;
}


ColorLegend::~ColorLegend()
{
    this->colorScale = nullptr;
}


void ColorLegend::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    if (this->colorScale == nullptr) return;
    if (isEqual(this->colorScale->minimum(), NODATA)
        || isEqual(this->colorScale->maximum(), NODATA)) return;

    QPainter painter(this);
    Crit3DColor* myColor;

    // clean widget
    painter.setBrush(Qt::white);
    painter.fillRect(0, 0, painter.window().width(), painter.window().height(), painter.brush());

    const int BLANK_DX = 16;
    int legendWidth = painter.window().width() - BLANK_DX*2;
    unsigned int nrStep = this->colorScale->nrColors();
    float step = (colorScale->maximum() - colorScale->minimum()) / float(nrStep);
    double dx = double(legendWidth) / double(nrStep+1);
    unsigned int stepText = MAXVALUE(nrStep / 4, 1);
    QString valueStr;
    int nrDigits;
    double dblValue, shiftFatctor;

    float value = this->colorScale->minimum();
    for (unsigned int i = 0; i <= nrStep; i++)
    {
        dblValue = double(value);
        myColor = this->colorScale->getColor(value);
        painter.setBrush(QColor(myColor->red, myColor->green, myColor->blue));
        painter.fillRect(int(BLANK_DX + dx*i +1), 0, int(ceil(dx)), 20, painter.brush());

        if ((i % stepText) == 0)
        {
            if (isEqual(dblValue, 0))
            {
                nrDigits = 1;
            }
            else
            {
                nrDigits = int(ceil(log10(abs(dblValue))));
            }

            // negative numbers
            if (dblValue < 0) nrDigits++;

            // integer numbers
            if (isEqual(round(dblValue), dblValue))
            {
                valueStr = QString::number(round(dblValue));
            }
            else
            {
                // one decimal
                if (fabs(round(dblValue*10) - (dblValue*10)) < 0.1)
                {
                    valueStr = QString::number(dblValue, 'f', 1);
                    nrDigits += 1;
                }
                // two decimals
                else
                {
                    valueStr = QString::number(dblValue, 'f', 2);
                    nrDigits += 2;
                }
            }

            shiftFatctor = 1.0 / nrDigits;
            painter.drawText(int(BLANK_DX*shiftFatctor + dx*i -1), 36, valueStr);
        }

        value += step;
    }
}
