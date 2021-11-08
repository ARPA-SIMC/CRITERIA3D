#ifndef PROXYWIDGET_H
#define PROXYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "meteoPoint.h"
    #include "interpolationSettings.h"

    class Crit3DProxyWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, QList<Crit3DMeteoPoint*> meteoPointList);
            ~Crit3DProxyWidget();

    private:
            Crit3DInterpolationSettings* interpolationSettings;
            QList<Crit3DMeteoPoint*> meteoPointList;
            QComboBox variable;
            QComboBox axisX;
            QCheckBox detrended;
            QCheckBox climatologyLR;
            QCheckBox modelLP;
            QCheckBox zeroIntercept;
            QTextEdit r2;
            QTextEdit lapseRate;
            QTextEdit r2ThermalLevels;
            
    signals:
    };


#endif // PROXYWIDGET_H
