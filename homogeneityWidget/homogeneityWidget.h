#ifndef HOMOGENEITYWIDGET_H
#define HOMOGENEITYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "homogeneityChartView.h"
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
            Crit3DHomogeneityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPoints, QDate firstDaily,
                                        QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality* quality);
            ~Crit3DHomogeneityWidget();
            void closeEvent(QCloseEvent *event);
            void changeMethod(const QString methodName);
            void changeVar(const QString varName);
            //void plot();
            void on_actionChangeLeftAxis();
            void on_actionExportGraph();
            void on_actionExportData();
            void addStationClicked();
            void deleteStationClicked();
            void saveToDbClicked();
            void updateYears();
            void setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint *meteoPointSet, QDate myDate);

    private:
            Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
            QList<Crit3DMeteoPoint> meteoPoints;
            Crit3DClimate clima;
            QList<std::string> idPoints;
            QDate firstDaily;
            QDate lastDaily;

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
            HomogeneityChartView *chartView;
            QComboBox jointStationsList;
            QPushButton addJointStation;
            QPushButton deleteJointStation;
            QPushButton saveToDb;
            QListWidget jointStationsSelected;
            QLineEdit minNumStations;
            QListWidget listFoundStations;
            QListWidget listSelectedStations;
            QPushButton addButton;
            QPushButton deleteButton;
            QTableWidget stationsTable;
            QLabel resultLabel;
            QPushButton execute;
    };


#endif // HOMOGENEITYWIDGET_H
