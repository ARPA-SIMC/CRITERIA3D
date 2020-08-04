/*
 * zoom functionality from: https://github.com/martonmiklos/qt_zoomable_chart_widget
*/

#include "rangelimitedvalueaxis.h"

RangeLimitedValueAxis::RangeLimitedValueAxis(QObject *parent) :
    QValueAxis(parent)
{
    setProperty("rangeLimited", true); // This is a hack
}

void RangeLimitedValueAxis::setLowerLimit(qreal value)
{
    if (m_limitLowerRange && value > m_upLimit) {
        m_limitLowerRange = false;
        return;
    }
    m_lowLimit = value;
    m_limitLowerRange = true;
}

void RangeLimitedValueAxis::setUpperLimit(qreal value)
{
    if (m_limitLowerRange && m_lowLimit > value) {
        m_limitUpperRange = false;
        return;
    }
    m_upLimit = value;
    m_limitUpperRange = true;
}

void RangeLimitedValueAxis::disableLowerLimit()
{
    m_limitLowerRange = false;
}

void RangeLimitedValueAxis::disableUpperLimit()
{
    m_limitUpperRange = false;
}

bool RangeLimitedValueAxis::isLowerRangeLimited() const
{
    return m_limitLowerRange;
}

bool RangeLimitedValueAxis::isUpperRangeLimited() const
{
    return m_limitUpperRange;
}

qreal RangeLimitedValueAxis::lowerLimit() const
{
    return m_lowLimit;
}

qreal RangeLimitedValueAxis::upperLimit() const
{
    return m_upLimit;
}
