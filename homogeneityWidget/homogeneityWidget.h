#ifndef HOMOGENEITYWIDGET_H
#define HOMOGENEITYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "homogeneityChartView.h"
    #include "annualSeriesChartView.h"
    #include "meteoPoint.h"
    #include "dbMeteoPointsHandler.h"
    #include "crit3dClimate.h"
    #include "interpolationSettings.h"
    #include "interpolationPoint.h"

    #define TOLERANCE 0.00000001

    class Crit3DHomogeneityWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DHomogeneityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPointsNearDistanceList, QList<std::string> sortedId,
            std::vector<float> distanceId, QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings,
                                    Crit3DClimateParameters *climateParameters, Crit3DQuality* quality);
            ~Crit3DHomogeneityWidget();
            void closeEvent(QCloseEvent *event);
            void changeMethod(const QString methodName);
            void changeVar(const QString varName);
            void changeYears();
            void plotAnnualSeries();
            void on_actionChangeLeftAxis();
            void on_actionExportHomogeneityGraph();
            void on_actionExportAnnualGraph();
            void on_actionExportAnnualData();
            void on_actionExportHomogeneityData();
            void addJointStationClicked();
            void deleteJointStationClicked();
            void saveToDbClicked();
            void updateYears();
            void setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint *meteoPointSet, QDate myDate);
            void findReferenceStations();
            void addFoundStationClicked();
            void deleteFoundStationClicked();
            void executeClicked();

    private:
            Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
            QList<Crit3DMeteoPoint> meteoPointsNearDistanceList;
            Crit3DClimate clima;
            QList<std::string> idPointsJointed;
            QDate firstDaily;
            QDate lastDaily;
            std::vector<float> myAnnualSeries;
            QList<std::string> sortedId;
            std::vector<float> distanceId;
            QMap<QString, std::string> mapNameId;
            QMap<QString, std::vector<float>> mapNameAnnualSeries;

            Crit3DMeteoSettings *meteoSettings;
            QSettings *settings;
            Crit3DClimateParameters *climateParameters;
            Crit3DQuality* quality;

            QComboBox variable;
            QComboBox method;
            QComboBox yearFrom;
            QComboBox yearTo;
            meteoVariable myVar;
            QPushButton find;
            HomogeneityChartView *homogeneityChartView;
            AnnualSeriesChartView *annualSeriesChartView;
            QComboBox jointStationsList;
            QPushButton addJointStation;
            QPushButton deleteJointStation;
            QPushButton saveToDb;
            QListWidget jointStationsSelected;
            QLineEdit minNumStations;
            QListWidget listFoundStations;
            QListWidget listSelectedStations;
            QPushButton addStationFoundButton;
            QPushButton deleteStationFoundButton;
            QTableWidget stationsTable;
            QLabel resultLabel;
            QPushButton execute;

            QString myError;
            double averageValue;
            float SNHT_T95_VALUES [10] {5.7,6.95,7.65,8.1,8.45,8.65,8.8,8.95,9.05,9.15};
    };


#endif // HOMOGENEITYWIDGET_H
