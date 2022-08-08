﻿#ifndef SYNCHRONICITYWIDGET_H
#define SYNCHRONICITYWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "synchronicityChartView.h"
    #include "interpolationChartView.h"
    #include "meteoPoint.h"
    #include "dbMeteoPointsHandler.h"
    #include "crit3dClimate.h"
    #include "interpolationSettings.h"
    #include "interpolationPoint.h"

    #define TOLERANCE 0.00000001

    class Crit3DSynchronicityWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DSynchronicityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoPoint mp, gis::Crit3DGisSettings gisSettings, QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings,
                                    Crit3DClimateParameters *climateParameters, Crit3DQuality* quality, Crit3DInterpolationSettings interpolationSettings, Crit3DInterpolationSettings qualityInterpolationSettings, bool checkSpatialQuality, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints);
            ~Crit3DSynchronicityWidget();
            void closeEvent(QCloseEvent *event);
            void changeVar(const QString varName);
            void changeYears();      
            void addStationGraph();
            void clearStationGraph();
            void addInterpolationGraph();
            void clearInterpolationGraph();
            void changeSmooth();
            void changeInterpolationDate();
            void smoothSerie();
            void setReferencePointId(const std::string &value);
            void on_actionChangeLeftSynchAxis();
            void on_actionChangeLeftInterpolationAxis();

    private:
            Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
            Crit3DClimate clima;
            Crit3DMeteoPoint mp;
            Crit3DMeteoPoint mpRef;
            std::string referencePointId;
            QDate firstDaily;
            QDate lastDaily;
            QDate firstRefDaily;
            QDate lastRefDaily;
            gis::Crit3DGisSettings gisSettings;
            Crit3DMeteoSettings *meteoSettings;
            QSettings *settings;
            Crit3DClimateParameters *climateParameters;
            Crit3DInterpolationSettings interpolationSettings;
            Crit3DInterpolationSettings qualityInterpolationSettings;
            bool checkSpatialQuality;
            Crit3DMeteoPoint* meteoPoints;
            int nrMeteoPoints;
            Crit3DQuality* quality;
            QLabel nameRefLabel;
            QComboBox variable;
            QComboBox stationYearFrom;
            QComboBox stationYearTo;
            QDateEdit interpolationDateFrom;
            QDateEdit interpolationDateTo;
            meteoVariable myVar;
            QSpinBox stationLag;
            QPushButton stationAddGraph;
            QPushButton stationClearGraph;
            QPushButton interpolationAddGraph;
            QComboBox interpolationElab;
            QPushButton interpolationClearGraph;
            QSpinBox interpolationLag;
            QSpinBox smooth;
            SynchronicityChartView *synchronicityChartView;
            InterpolationChartView *interpolationChartView;
            std::vector<float> interpolationDailySeries;
            std::vector<float> smoothInterpDailySeries;
            QDate interpolationStartDate;
            bool stationClearAndReload;
            bool interpolationClearAndReload;
            bool interpolationChangeSmooth;

    signals:
            void closeSynchWidget();
    };


#endif // SYNCHRONICITYWIDGET_H
