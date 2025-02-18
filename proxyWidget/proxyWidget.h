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
            Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality* quality,  Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *climateParam, bool checkSpatialQuality, int macroAreaNumber);
            ~Crit3DProxyWidget() override;
            void closeEvent(QCloseEvent *event) override;
            void updateDateTime(QDate newDate, int newHour);
            void updateFrequency(frequencyType newFrequency);
            void changeProxyPos(const QString proxyName);
            void changeVar(const QString varName);
            void plot();
            void climatologicalLRClicked(int toggled);
            void modelLRClicked(int toggled);
            void addMacroAreaLR();

    private:

            int nrMeteoPoints;
            bool checkSpatialQuality;
            int currentHour;
            int proxyPos;
            int macroAreaNumber;

            std::vector <Crit3DInterpolationDataPoint> outInterpolationPoints;
            Crit3DInterpolationSettings* interpolationSettings;
            Crit3DQuality* quality;
            Crit3DInterpolationSettings* SQinterpolationSettings;
            Crit3DMeteoSettings *meteoSettings;
            Crit3DMeteoPoint* meteoPoints;
            Crit3DClimateParameters *climateParam;

            frequencyType currentFrequency;
            QDate currentDate;

            QValueAxis *axisY_sx;
            QComboBox comboVariable;
            QComboBox comboAxisX;
            QCheckBox detrended;
            QCheckBox climatologicalLR;
            QCheckBox modelLR;
            QTextEdit r2;
            QTextEdit lapseRate;
            ChartView *chartView;
            meteoVariable myVar;

            Crit3DTime getCurrentTime();

            void on_actionChangeLeftAxis();

    signals:
        void closeProxyWidget();
    };


#endif // PROXYWIDGET_H
