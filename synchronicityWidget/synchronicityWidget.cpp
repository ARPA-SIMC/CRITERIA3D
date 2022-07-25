/*!
    \copyright 2020 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of AGROLIB.
    AGROLIB has been developed under contract issued by ARPAE Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include "meteo.h"
#include "synchronicityWidget.h"
#include "synchronicityChartView.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "climate.h"
#include "dialogChangeAxis.h"
#include "gammaFunction.h"
#include "furtherMathFunctions.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DSynchronicityWidget::Crit3DSynchronicityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoPoint* mp,
                                                         QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:meteoPointsDbHandler(meteoPointsDbHandler), mp(mp),firstDaily(firstDaily),
  lastDaily(lastDaily), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Synchronicity analysis Id:"+QString::fromStdString(mp->id));
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *upperLayout = new QHBoxLayout();
    QVBoxLayout *plotLayout = new QVBoxLayout();

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

    QLabel *nameLabel = new QLabel(QString::fromStdString(mp->name));
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
    interpolationYearFrom.setMaximumWidth(100);
    interpolationDateLayout->addWidget(&interpolationYearFrom);
    QLabel *interpolationYearToLabel = new QLabel(tr("From"));
    interpolationDateLayout->addWidget(interpolationYearToLabel);
    interpolationYearTo.setMaximumWidth(100);
    interpolationDateLayout->addWidget(&interpolationYearTo);
    for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
    {
        interpolationYearFrom.addItem(QString::number(firstDaily.year()+i));
        interpolationYearTo.addItem(QString::number(firstDaily.year()+i));
    }
    interpolationYearTo.setCurrentText(QString::number(lastDaily.year()));
    QLabel *interpolationLagLabel = new QLabel(tr("lag"));
    interpolationDateLayout->addStretch(20);
    interpolationDateLayout->addWidget(interpolationLagLabel);
    interpolationLag.setRange(-10, 10);
    interpolationLag.setSingleStep(1);
    interpolationDateLayout->addWidget(&interpolationLag);
    interpolationLayout->addLayout(interpolationDateLayout);

    interpolationButtonLayout->addWidget(&interpolationAddGraph);
    interpolationAddGraph.setText("Add graph");
    interpolationButtonLayout->addWidget(&interpolationReloadGraph);
    interpolationReloadGraph.setText("Reload");
    interpolationButtonLayout->addWidget(&interpolationClearGraph);
    interpolationClearGraph.setText("Clear");
    interpolationLayout->addLayout(interpolationButtonLayout);
    interpolationGroupBox->setLayout(interpolationLayout);

    synchronicityChartView = new SynchronicityChartView();
    synchronicityChartView->setMinimumWidth(this->width()*2/3);
    plotLayout->addWidget(synchronicityChartView);

    upperLayout->addWidget(firstGroupBox);
    upperLayout->addWidget(stationGroupBox);
    upperLayout->addWidget(interpolationGroupBox);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    QAction* changeSynchronicityLeftAxis = new QAction(tr("&Change synchronicity chart axis left"), this);

    editMenu->addAction(changeSynchronicityLeftAxis);

    mainLayout->addLayout(upperLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&stationYearFrom, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&stationYearTo, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&stationAddGraph, &QPushButton::clicked, [=](){ addGraph(); });
    connect(changeSynchronicityLeftAxis, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionChangeLeftAxis);

    show();
}


Crit3DSynchronicityWidget::~Crit3DSynchronicityWidget()
{

}

void Crit3DSynchronicityWidget::closeEvent(QCloseEvent *event)
{
    event->accept();
}

void Crit3DSynchronicityWidget::setReferencePointId(const std::string &value)
{
    referencePointId = value;
    nameRefLabel.setText(QString::fromStdString(referencePointId));
}


void Crit3DSynchronicityWidget::changeVar(const QString varName)
{
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    // TO DO
}

void Crit3DSynchronicityWidget::changeYears()
{
    // TO DO
}

void Crit3DSynchronicityWidget::addGraph()
{
    if (referencePointId == "")
    {
        QMessageBox::information(nullptr, "Error", "Select a reference point on the map");
        return;
    }
    // TO DO
}

void Crit3DSynchronicityWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(true);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        synchronicityChartView->setYmax(changeAxisDialog.getMaxVal());
        synchronicityChartView->setYmin(changeAxisDialog.getMinVal());
    }
}


