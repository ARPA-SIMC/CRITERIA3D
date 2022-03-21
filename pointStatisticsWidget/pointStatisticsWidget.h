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
            Crit3DPointStatisticsWidget(bool isGrid, QList<Crit3DMeteoPoint> meteoPoints);
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
            bool isGrid;
            QList<Crit3DMeteoPoint> meteoPoints;
            frequencyType currentFrequency;
            QComboBox variable;
            QComboBox yearFrom;
            QComboBox yearTo;
            meteoVariable myVar;
            QPushButton elaboration;
            QDateEdit dayFrom;
            QDateEdit dayTo;
            QSpinBox hour;
            QPushButton compute;
            PointStatisticsChartView *chartView;
            QComboBox jointStationsList;
            QPushButton addStation;
            QPushButton deleteStation;
            QPushButton saveToDb;
            QListWidget jointStationsSelected;
            QComboBox graph;
            QTextEdit availability;
            QTextEdit rate;
            QTextEdit r2;
            QTextEdit significance;
            QTextEdit average;
            QTextEdit mode;
            QTextEdit median;
            QTextEdit sigma;
            QTextEdit classWidth;
            QTextEdit valMax;
            QTextEdit smoothing;

    signals:
        void closePointStatistics();
    };


#endif // POINTSTATISTICSWIDGET_H
