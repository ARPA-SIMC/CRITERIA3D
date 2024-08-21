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


    class Crit3DHomogeneityWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DHomogeneityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPointsNearDistanceList, QList<std::string> sortedId,
            std::vector<float> distanceId, QList<QString> jointStationsMyMp, QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings,
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
            void findReferenceStations();
            void addFoundStationClicked();
            void deleteFoundStationClicked();
            void executeClicked();
            void checkValueAndMerge(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint* meteoPointSet, QDate myDate);

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
            QList<QString> listAllFound;
            QPushButton addStationFoundButton;
            QPushButton deleteStationFoundButton;
            QTableWidget stationsTable;
            QLabel resultLabel;
            QPushButton execute;

            QString myError;
            double averageValue;
            float SNHT_T95_VALUES [10] {5.7f,6.95f,7.65f,8.1f,8.45f,8.65f,8.8f,8.95f,9.05f,9.15f};
    };


#endif // HOMOGENEITYWIDGET_H
