#ifndef PROXYWIDGET_H
#define PROXYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "chartView.h"
    #include "meteoPoint.h"
    #include "interpolationSettings.h"
    #include "interpolationPoint.h"

    class Crit3DProxyWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality* quality,  Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *climateParam, bool checkSpatialQuality);
            ~Crit3DProxyWidget();
            void closeEvent(QCloseEvent *event);
            void updateDateTime(QDate newDate, int newHour);
            void updateFrequency(frequencyType newFrequency);
            void changeProxyPos(const QString proxyName);
            void changeVar(const QString varName);
            void plot();
            void climatologicalLRClicked(int toggled);
            void modelLRClicked(int toggled);

    private:
            Crit3DInterpolationSettings* interpolationSettings;
            Crit3DQuality* quality;
            Crit3DInterpolationSettings* SQinterpolationSettings;
            Crit3DMeteoSettings *meteoSettings;
            Crit3DMeteoPoint* meteoPoints;
            Crit3DClimateParameters *climateParam;
            int nrMeteoPoints;
            bool checkSpatialQuality;
            frequencyType currentFrequency;
            QDate currentDate;
            int currentHour;
            std::vector <Crit3DInterpolationDataPoint> outInterpolationPoints;
            QComboBox comboVariable;
            QComboBox comboAxisX;
            QCheckBox detrended;
            QCheckBox climatologicalLR;
            QCheckBox modelLR;
            QTextEdit r2;
            QTextEdit lapseRate;
            ChartView *chartView;
            meteoVariable myVar;
            int proxyPos;

            Crit3DTime getCurrentTime();

    signals:
        void closeProxyWidget();
    };


#endif // PROXYWIDGET_H
