/*!
    CRITERIA3D
    \copyright 2016 Fausto Tomei, Gabriele Antolini, Laura Costantini
    Alberto Pistocchi, Marco Bittelli, Antonio Volta
    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna
    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the /NU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.
    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/


#include "proxyWidget.h"
#include "utilities.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DProxyWidget::Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints)
:interpolationSettings(interpolationSettings), meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints)
{
    
    this->setWindowTitle("Statistics");
    this->resize(1240, 700);
    this->setAttribute(Qt::WA_DeleteOnClose);
    

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGroupBox *horizontalGroupBox = new QGroupBox();
    QHBoxLayout *selectionLayout = new QHBoxLayout;
    QHBoxLayout *selectionChartLayout = new QHBoxLayout;
    QVBoxLayout *selectionOptionLayout = new QVBoxLayout;
    QHBoxLayout *selectionOptionBoxLayout = new QHBoxLayout;
    QHBoxLayout *selectionOptionEditLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    detrended.setText("Detrended data");
    climatologyLR.setText("Climatology lapse rate");
    modelLP.setText("Model lapse rate");
    zeroIntercept.setText("Zero intercept");
    
    QLabel *r2Label = new QLabel(tr("R2"));
    QLabel *lapseRateLabel = new QLabel(tr("Lapse rate"));
    QLabel *r2ThermalLevelsLabel = new QLabel(tr("R2 thermal levels"));
    
    r2.setMaximumWidth(50);
    r2.setMaximumHeight(30);
    lapseRate.setMaximumWidth(50);
    lapseRate.setMaximumHeight(30);
    r2ThermalLevels.setMaximumWidth(50);
    r2ThermalLevels.setMaximumHeight(30);
    
    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));
    
    selectionChartLayout->addWidget(variableLabel);
    selectionChartLayout->addWidget(&variable);
    selectionChartLayout->addWidget(axisXLabel);
    selectionChartLayout->addWidget(&axisX);
    
    selectionOptionBoxLayout->addWidget(&detrended);
    selectionOptionBoxLayout->addWidget(&climatologyLR);
    selectionOptionBoxLayout->addWidget(&modelLP);
    selectionOptionBoxLayout->addWidget(&zeroIntercept);

    selectionOptionEditLayout->addWidget(r2Label);
    selectionOptionEditLayout->addWidget(&r2);
    selectionOptionEditLayout->addSpacing(200);
    selectionOptionEditLayout->addWidget(lapseRateLabel);
    selectionOptionEditLayout->addWidget(&lapseRate);
    selectionOptionEditLayout->addSpacing(200);
    selectionOptionEditLayout->addWidget(r2ThermalLevelsLabel);
    selectionOptionEditLayout->addWidget(&r2ThermalLevels);
    selectionOptionEditLayout->addSpacing(200);

    selectionOptionLayout->addLayout(selectionOptionBoxLayout);
    selectionOptionLayout->addLayout(selectionOptionEditLayout);

    selectionLayout->addLayout(selectionChartLayout);
    selectionLayout->addSpacing(50);
    selectionLayout->addLayout(selectionOptionLayout);
    

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    /*
    QAction* changeLeftAxis = new QAction(tr("&Change axis left"), this);
    QAction* changeRightAxis = new QAction(tr("&Change axis right"), this);
    QAction* exportGraph = new QAction(tr("&Export graph"), this);

    editMenu->addAction(changeLeftAxis);
    editMenu->addAction(changeRightAxis);
    editMenu->addAction(exportGraph);
    */

    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);
    plotLayout->addWidget(chartView);

    horizontalGroupBox->setMaximumSize(1240, 130);
    horizontalGroupBox->setLayout(selectionLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    show();

}

Crit3DProxyWidget::~Crit3DProxyWidget()
{

}




