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

Crit3DProxyWidget::Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, QList<Crit3DMeteoPoint*> meteoPointList)
:interpolationSettings(interpolationSettings), meteoPointList(meteoPointList)
{
    
    this->setWindowTitle("Statistics");
    this->resize(1240, 700);
    this->setAttribute(Qt::WA_DeleteOnClose);
    

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGroupBox *horizontalGroupBox = new QGroupBox();
    QGridLayout *selectionLayout = new QGridLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    detrended.setText("Detrended data");
    climatologyLR.setText("Climatology lapse rate");
    modelLP.setText("Model lapse rate");
    zeroIntercept.setText("Zero intercept");
    
    QLabel *r2Label = new QLabel(tr("R2"));
    QLabel *lapseRateLabel = new QLabel(tr("Lapse rate"));
    QLabel *r2ThermalLevelsLabel = new QLabel(tr("R2 thermal levels"));
    
    r2.setMaximumWidth(100);
    r2.setMaximumHeight(30);
    lapseRate.setMaximumWidth(100);
    lapseRate.setMaximumHeight(30);
    r2ThermalLevels.setMaximumWidth(100);
    r2ThermalLevels.setMaximumHeight(30);
    
    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));
    
    selectionLayout->addWidget(variableLabel,0,0);
    selectionLayout->addWidget(&variable,1,0);
    selectionLayout->addWidget(axisXLabel,0,1);
    selectionLayout->addWidget(&axisX,1,1);
    
    selectionLayout->addWidget(&detrended,0,2);
    selectionLayout->addWidget(&climatologyLR,0,3);
    selectionLayout->addWidget(&modelLP,0,4);
    selectionLayout->addWidget(&zeroIntercept,1,2);
    selectionLayout->addWidget(r2Label,1,3);
    selectionLayout->addWidget(&r2,1,4);
    selectionLayout->addWidget(lapseRateLabel,2,1);
    selectionLayout->addWidget(&lapseRate,2,2);
    selectionLayout->addWidget(r2ThermalLevelsLabel,2,3);
    selectionLayout->addWidget(&r2ThermalLevels,2,4);
    

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


//    plotLayout->addWidget(chartView);
    horizontalGroupBox->setLayout(selectionLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

}

Crit3DProxyWidget::~Crit3DProxyWidget()
{

}




