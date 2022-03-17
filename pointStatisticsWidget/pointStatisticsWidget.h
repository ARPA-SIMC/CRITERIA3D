#ifndef POINTSTATISTICSWIDGET_H
#define POINTSTATISTICSWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "pointStatisticsChartView.h"
    #include "meteoPoint.h"
    #include "interpolationSettings.h"
    #include "interpolationPoint.h"

    class Crit3DPointStatisticsWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DPointStatisticsWidget(QList<Crit3DMeteoPoint> meteoPoints);
            ~Crit3DPointStatisticsWidget();
            void closeEvent(QCloseEvent *event);
            void updateDateTime(QDate newDate, int newHour);
            void updateFrequency(frequencyType newFrequency);
            void changeProxyPos(const QString proxyName);
            void changeVar(const QString varName);
            void plot();
            void climatologicalLRClicked(int toggled);
            void modelLRClicked(int toggled);

    private:
            QList<Crit3DMeteoPoint> meteoPoints;
            /*
            frequencyType currentFrequency;
            QDate currentDate;
            int currentHour;
            QComboBox variable;
            QComboBox axisX;
            QCheckBox detrended;
            QCheckBox climatologicalLR;
            QCheckBox modelLR;
            QTextEdit r2;
            QTextEdit lapseRate;
            PointStatisticsChartView *chartView;
            meteoVariable myVar;
            int proxyPos;
            */

    signals:
        void closePointStatistics();
    };


#endif // POINTSTATISTICSWIDGET_H
