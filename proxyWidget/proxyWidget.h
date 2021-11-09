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
            Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, frequencyType currentFrequency, QDateTime currentDateTime);
            ~Crit3DProxyWidget();
            void closeEvent(QCloseEvent *event);
            void updateDateTime(QDateTime newDateTime);
            void updateFrequency(frequencyType newFrequency);

    private:
            Crit3DInterpolationSettings* interpolationSettings;
            Crit3DMeteoPoint* meteoPoints;
            int nrMeteoPoints;
            frequencyType currentFrequency;
            QDateTime currentDateTime;
            QComboBox variable;
            QComboBox axisX;
            QCheckBox detrended;
            QCheckBox climatologyLR;
            QCheckBox modelLP;
            QCheckBox zeroIntercept;
            QTextEdit r2;
            QTextEdit lapseRate;
            QTextEdit r2ThermalLevels;
            QChartView *chartView;
            QChart *chart;
            
    signals:
        void closeProxyWidget();
    };


#endif // PROXYWIDGET_H
