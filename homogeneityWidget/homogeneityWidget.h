#ifndef HOMOGENEITYWIDGET_H
#define HOMOGENEITYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "homogeneityChartView.h"
    #include "annualSeriesChartView.h"
    #include "meteoPoint.h"
    #include "dbMeteoPointsHandler.h"
    #include "dbMeteoGrid.h"
    #include "crit3dClimate.h"
    #include "interpolationSettings.h"
    #include "interpolationPoint.h"

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
            void on_actionExportGraph();
            void on_actionExportData();
            void addJointStationClicked();
            void deleteJointStationClicked();
            void saveToDbClicked();
            void updateYears();
            void setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint *meteoPointSet, QDate myDate);
            void findReferenceStations();
            void addFoundStationClicked();
            void deleteFoundStationClicked();

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
            QList<std::string> sortedIdFound;
            QList<float> distanceIdFound;
            QList<std::vector<float>> myAnnualSeriesFound;

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
    };


#endif // HOMOGENEITYWIDGET_H
