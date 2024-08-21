#ifndef SYNCHRONICITYWIDGET_H
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


    class Crit3DSynchronicityWidget : public QWidget
    {
        Q_OBJECT

        public:
        Crit3DSynchronicityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoPoint mp, gis::Crit3DGisSettings gisSettings,
                                  QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings,
                                  Crit3DClimateParameters *climateParameters, Crit3DQuality* quality,
                                  Crit3DInterpolationSettings interpolationSettings,
                                  Crit3DInterpolationSettings qualityInterpolationSettings,
                                  bool checkSpatialQuality, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints)
            :meteoPointsDbHandler(meteoPointsDbHandler), mp(mp),firstDaily(firstDaily), lastDaily(lastDaily),
              gisSettings(gisSettings), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters),
              interpolationSettings(interpolationSettings), qualityInterpolationSettings(qualityInterpolationSettings),
              checkSpatialQuality(checkSpatialQuality), meteoPoints(meteoPoints),
              nrMeteoPoints(nrMeteoPoints), quality(quality)
        {
            this->setWindowTitle("Synchronicity analysis Id:"+QString::fromStdString(mp.id));
            this->resize(1240, 700);
            this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            this->setAttribute(Qt::WA_DeleteOnClose);

            stationClearAndReload = false;
            interpolationClearAndReload = false;
            interpolationChangeSmooth = false;

            // layout
            QVBoxLayout *mainLayout = new QVBoxLayout();
            QHBoxLayout *upperLayout = new QHBoxLayout();
            QHBoxLayout *plotLayout = new QHBoxLayout();

            QGroupBox *firstGroupBox = new QGroupBox();
            QVBoxLayout *firstLayout = new QVBoxLayout();
            QGroupBox *stationGroupBox = new QGroupBox();
            QVBoxLayout *stationLayout = new QVBoxLayout;
            QHBoxLayout *stationRefPointLayout = new QHBoxLayout;
            QHBoxLayout *stationDateLayout = new QHBoxLayout;
            QHBoxLayout *stationButtonLayout = new QHBoxLayout;
            QGroupBox *interpolationGroupBox = new QGroupBox();
            QVBoxLayout *interpolationLayout = new QVBoxLayout;
            QHBoxLayout *interpolationDateLayout = new QHBoxLayout;
            QHBoxLayout *interpolationButtonLayout = new QHBoxLayout;
            referencePointId = "";

            QLabel *nameLabel = new QLabel(QString::fromStdString(mp.name));
            QLabel *variableLabel = new QLabel(tr("Variable: "));
            variable.addItem("DAILY_TMIN");
            variable.addItem("DAILY_TMAX");
            variable.addItem("DAILY_PREC");

            myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
            variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
            variable.setMaximumWidth(150);
            firstLayout->addWidget(nameLabel);
            firstLayout->addWidget(variableLabel);
            firstLayout->addWidget(&variable);
            firstGroupBox->setLayout(firstLayout);

            stationGroupBox->setTitle("Station");
            QLabel *referenceLabel = new QLabel(tr("Reference point: "));
            stationRefPointLayout->addWidget(referenceLabel);
            stationRefPointLayout->addWidget(&nameRefLabel);
            QLabel *yearFromLabel = new QLabel(tr("From"));
            stationDateLayout->addWidget(yearFromLabel);
            stationYearFrom.setMaximumWidth(100);
            stationDateLayout->addWidget(&stationYearFrom);
            QLabel *yearToLabel = new QLabel(tr("To"));
            stationDateLayout->addWidget(yearToLabel);
            stationYearTo.setMaximumWidth(100);
            stationDateLayout->addWidget(&stationYearTo);
            for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
            {
                stationYearFrom.addItem(QString::number(firstDaily.year()+i));
                stationYearTo.addItem(QString::number(firstDaily.year()+i));
            }
            stationYearTo.setCurrentText(QString::number(lastDaily.year()));
            QLabel *lagLabel = new QLabel(tr("lag"));
            stationDateLayout->addStretch(20);
            stationDateLayout->addWidget(lagLabel);
            stationLag.setRange(-10, 10);
            stationLag.setSingleStep(1);
            stationDateLayout->addWidget(&stationLag);
            stationLayout->addLayout(stationRefPointLayout);
            stationLayout->addLayout(stationDateLayout);

            stationButtonLayout->addWidget(&stationAddGraph);
            stationAddGraph.setText("Add graph");
            stationButtonLayout->addWidget(&stationClearGraph);
            stationClearGraph.setText("Clear");
            stationLayout->addLayout(stationButtonLayout);
            stationGroupBox->setLayout(stationLayout);

            interpolationGroupBox->setTitle("Interpolation");
            QLabel *interpolationYearFromLabel = new QLabel(tr("From"));
            interpolationDateLayout->addWidget(interpolationYearFromLabel);
            interpolationDateFrom.setMaximumWidth(100);
            interpolationDateLayout->addWidget(&interpolationDateFrom);
            QLabel *interpolationYearToLabel = new QLabel(tr("To"));
            interpolationDateLayout->addWidget(interpolationYearToLabel);
            interpolationDateTo.setMaximumWidth(100);
            interpolationDateLayout->addWidget(&interpolationDateTo);
            interpolationDateFrom.setMinimumDate(firstDaily);
            interpolationDateTo.setMinimumDate(firstDaily);
            interpolationDateFrom.setMaximumDate(lastDaily);
            interpolationDateTo.setMaximumDate(lastDaily);
            interpolationDateFrom.setDate(firstDaily);
            interpolationDateTo.setDate(lastDaily);
            interpolationDateLayout->addStretch(20);
            QLabel *interpolationLagLabel = new QLabel(tr("lag"));
            interpolationDateLayout->addWidget(interpolationLagLabel);
            interpolationLag.setRange(-10, 10);
            interpolationLag.setSingleStep(1);
            interpolationDateLayout->addWidget(&interpolationLag);
            QLabel *smoothLabel = new QLabel(tr("smooth"));
            interpolationDateLayout->addWidget(smoothLabel);
            smooth.setRange(0, 10);
            smooth.setSingleStep(1);
            interpolationDateLayout->addWidget(&smooth);
            interpolationLayout->addLayout(interpolationDateLayout);

            QLabel *interpolationElabLabel = new QLabel(tr("Elaboration: "));
            interpolationElab.addItem("Difference");
            interpolationElab.addItem("Absolute difference");
            interpolationElab.addItem("Square difference");
            interpolationButtonLayout->addWidget(interpolationElabLabel);
            interpolationButtonLayout->addWidget(&interpolationElab);
            interpolationButtonLayout->addWidget(&interpolationAddGraph);
            interpolationAddGraph.setText("Add graph");
            interpolationButtonLayout->addWidget(&interpolationClearGraph);
            interpolationClearGraph.setText("Clear");
            interpolationLayout->addLayout(interpolationButtonLayout);
            interpolationGroupBox->setLayout(interpolationLayout);

            synchronicityChartView = new SynchronicityChartView();
            synchronicityChartView->setMinimumWidth(this->width()*2/3);
            plotLayout->addWidget(synchronicityChartView);

            interpolationChartView = new InterpolationChartView();
            interpolationChartView->setMinimumWidth(this->width()*2/3);
            interpolationChartView->setVisible(false);
            plotLayout->addWidget(interpolationChartView);

            upperLayout->addWidget(firstGroupBox);
            upperLayout->addWidget(stationGroupBox);
            upperLayout->addWidget(interpolationGroupBox);

            // menu
            QMenuBar* menuBar = new QMenuBar();
            QMenu *editMenu = new QMenu("Edit");

            menuBar->addMenu(editMenu);
            mainLayout->setMenuBar(menuBar);

            QAction* changeSynchronicityLeftAxis = new QAction(tr("&Change synchronicity chart axis left"), this);
            QAction* changeInterpolationLeftAxis = new QAction(tr("&Change interpolation chart axis left"), this);

            editMenu->addAction(changeSynchronicityLeftAxis);
            editMenu->addAction(changeInterpolationLeftAxis);

            mainLayout->addLayout(upperLayout);
            mainLayout->addLayout(plotLayout);
            setLayout(mainLayout);

            connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
            connect(&stationYearFrom, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
            connect(&stationYearTo, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
            connect(&stationAddGraph, &QPushButton::clicked, [=](){ addStationGraph(); });
            connect(&stationClearGraph, &QPushButton::clicked, [=](){ clearStationGraph(); });
            connect(&interpolationAddGraph, &QPushButton::clicked, [=](){ addInterpolationGraph(); });
            connect(&interpolationClearGraph, &QPushButton::clicked, [=](){ clearInterpolationGraph(); });
            connect(&smooth, QOverload<int>::of(&QSpinBox::valueChanged), [=](){ changeSmooth(); });
            connect(&interpolationDateFrom, &QDateEdit::dateChanged, [=](){ this->changeInterpolationDate(); });
            connect(&interpolationDateTo, &QDateEdit::dateChanged, [=](){ this->changeInterpolationDate(); });
            connect(changeSynchronicityLeftAxis, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionChangeLeftSynchAxis);
            connect(changeInterpolationLeftAxis, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionChangeLeftInterpolationAxis);

            show();
        }
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
            void smoothSeries();
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
