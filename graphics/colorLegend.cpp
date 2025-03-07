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

    // check
    if (this->colorScale == nullptr)
        return;
    if (isEqual(this->colorScale->minimum(), NODATA)
        || isEqual(this->colorScale->maximum(), NODATA))
        return;

    QPainter painter(this);
    Crit3DColor* myColor;

    // clean widget
    painter.setBrush(Qt::white);
    painter.fillRect(0, 0, painter.window().width(), painter.window().height(), painter.brush());

    const int BLANK_DX = 16;
    int legendWidth = painter.window().width() - BLANK_DX*2;

    unsigned int nrStep = this->colorScale->nrColors();
    unsigned int nrStepText = MAXVALUE(round(float(nrStep) / 4.f), 1);

    double dx = double(legendWidth) / double(nrStep+1);

    double value = this->colorScale->minimum();
    double step = (colorScale->maximum() - colorScale->minimum()) / double(nrStep);
    double range = (colorScale->maximum() - colorScale->minimum());

    QString valueStr;
    int nrDigits;
    for (unsigned int i = 0; i <= nrStep; i++)
    {
        myColor = this->colorScale->getColor(value);
        painter.setBrush(QColor(myColor->red, myColor->green, myColor->blue));
        painter.fillRect(int(BLANK_DX + dx*i +1), 0, int(ceil(dx)), 20, painter.brush());

        if ((i % nrStepText) == 0)
        {
            if (fabs(value) <= 1)
            {
                nrDigits = 1;
            }
            else
            {
                nrDigits = int(ceil(log10(fabs(value))));
            }

            // negative numbers
            if (value < 0) nrDigits++;

            double decimal = fabs(value - round(value));
            if ((decimal / range) > 0.1)
            {
                // two decimals
                valueStr = QString::number(value, 'f', 2);
                nrDigits += 2;
            }
            else
            {
                if ((decimal / range) > 0.01)
                {
                    // one decimal
                    valueStr = QString::number(value, 'f', 1);
                    nrDigits += 1;
                }
                else
                {
                    // integer
                    valueStr = QString::number(round(value));
                }
            }

            double shiftFatctor = 1. / double(nrDigits);
            painter.drawText(int(BLANK_DX*shiftFatctor + dx*i -1), 36, valueStr);
        }

        value += step;
    }
}
