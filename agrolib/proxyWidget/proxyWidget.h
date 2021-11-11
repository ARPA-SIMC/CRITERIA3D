#ifndef PROXYWIDGET_H
#define PROXYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "chartView.h"
    #include "meteoPoint.h"
    #include "interpolationSettings.h"

    class Crit3DProxyWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, QList<Crit3DMeteoPoint> &primaryList, QList<Crit3DMeteoPoint> &supplementalList, QList<Crit3DMeteoPoint> &secondaryList, frequencyType currentFrequency, QDate currentDate, int currentHour);
            ~Crit3DProxyWidget();
            void closeEvent(QCloseEvent *event);
            void updateDateTime(QDate newDate, int newHour);
            void updateFrequency(frequencyType newFrequency);
            void changeProxyPos(const QString proxyName);
            void changeVar(const QString varName);
            void plot();
            void climatologicalLRClicked(int toggled);
            void computeHighestStationIndex();
            void updatePointList(const QList<Crit3DMeteoPoint> &primaryValue, const QList<Crit3DMeteoPoint> &secondaryValue, const QList<Crit3DMeteoPoint> &supplementalValue );

    private:
            Crit3DInterpolationSettings* interpolationSettings;
            QList<Crit3DMeteoPoint> primaryList;
            QList<Crit3DMeteoPoint> secondaryList;
            QList<Crit3DMeteoPoint> supplementalList;
            frequencyType currentFrequency;
            QDate currentDate;
            int currentHour;
            QComboBox variable;
            QComboBox axisX;
            QCheckBox detrended;
            QCheckBox climatologicalLR;
            QCheckBox modelLP;
            QCheckBox zeroIntercept;
            QTextEdit r2;
            QTextEdit lapseRate;
            QTextEdit r2ThermalLevels;
            ChartView *chartView;
            meteoVariable myVar;
            int proxyPos;

            int highestStationIndex;
            double zMax;
            int highestStationBelongToList;

    signals:
        void closeProxyWidget();
    };


#endif // PROXYWIDGET_H
