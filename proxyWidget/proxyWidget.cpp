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

#include "meteo.h"
#include "proxyWidget.h"
#include "utilities.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DProxyWidget::Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, QList<Crit3DMeteoPoint> &primaryList, QList<Crit3DMeteoPoint> &secondaryList, QList<Crit3DMeteoPoint> &supplementalList, frequencyType currentFrequency, QDateTime currentDateTime)
:interpolationSettings(interpolationSettings), primaryList(primaryList), secondaryList(secondaryList), supplementalList(supplementalList), currentFrequency(currentFrequency), currentDateTime(currentDateTime)
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
    climatologicalLR.setText("Climatological lapse rate");
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

    std::vector<Crit3DProxy> proxy = interpolationSettings->getCurrentProxy();

    for(int i=0; i<proxy.size(); i++)
    {
        axisX.addItem(QString::fromStdString(proxy[i].getName()));
    }
    proxyPos = 0;
    axisX.setSizeAdjustPolicy(QComboBox::AdjustToContents);

    std::map<meteoVariable, std::string>::const_iterator it;
    if (currentFrequency == daily)
    {
        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.currentText().toStdString());
    }
    variable.setMinimumWidth(130);
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    
    selectionChartLayout->addWidget(variableLabel);
    selectionChartLayout->addWidget(&variable);
    selectionChartLayout->addWidget(axisXLabel);
    selectionChartLayout->addWidget(&axisX);
    
    selectionOptionBoxLayout->addWidget(&detrended);
    selectionOptionBoxLayout->addWidget(&climatologicalLR);
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
    
    connect(&axisX, &QComboBox::currentTextChanged, [=](const QString &newProxy){ this->changeProxyPos(newProxy); });
    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&climatologicalLR, &QCheckBox::toggled, [=](int toggled){ this->climatologicalLRClicked(toggled); });

    // compute highest station index
    computeHighestStationIndex();
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

    chartView = new ChartView();
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

void Crit3DProxyWidget::changeProxyPos(const QString proxyName)
{
    for (int pos=0; pos < interpolationSettings->getProxyNr(); pos++)
    {
        QString myProxy = QString::fromStdString(interpolationSettings->getProxy(pos)->getName());
        if (myProxy == proxyName)
        {
            proxyPos = pos;
            break;
        }
    }
    plot();
}

void Crit3DProxyWidget::changeVar(const QString varName)
{
    if (currentFrequency == daily)
    {
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    }
    else if (currentFrequency == hourly)
    {
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, varName.toStdString());
    }
    plot();
}

void Crit3DProxyWidget::updateDateTime(QDateTime newDateTime)
{
    currentDateTime = newDateTime;
    plot();
}

void Crit3DProxyWidget::updateFrequency(frequencyType newFrequency)
{
    if (newFrequency != currentFrequency)
    {
        currentFrequency = newFrequency;
        variable.clear();

        std::map<meteoVariable, std::string>::const_iterator it;
        if (currentFrequency == daily)
        {
            for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
            {
                variable.addItem(QString::fromStdString(it->second));
            }
            myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
        }
        else if (currentFrequency == hourly)
        {
            for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
            {
                variable.addItem(QString::fromStdString(it->second));
            }
            myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.currentText().toStdString());
        }
        variable.adjustSize();
    }

    plot();
}

void Crit3DProxyWidget::closeEvent(QCloseEvent *event)
{
    emit closeProxyWidget();
    event->accept();

}

void Crit3DProxyWidget::plot()
{
    QList<QPointF> point_vector;
    QList<QPointF> point_vector2;
    QList<QPointF> point_vector3;
    QPointF point;

    for (int i = 0; i<primaryList.size(); i++)
    {
        float proxyVal = primaryList[i].getProxyValue(proxyPos);
        float varVal;
        if (currentFrequency == daily)
        {
            varVal = primaryList[i].getMeteoPointValueD(getCrit3DDate(currentDateTime.date()), myVar);
        }
        else if (currentFrequency == hourly)
        {
            varVal = primaryList[i].getMeteoPointValueH(getCrit3DDate(currentDateTime.date()), currentDateTime.time().hour(), currentDateTime.time().minute(), myVar);
        }
        if (proxyVal != NODATA && varVal != NODATA)
        {
            point.setX(proxyVal);
            point.setY(varVal);
            point_vector.append(point);
        }
    }
    for (int i = 0; i<secondaryList.size(); i++)
    {
        float proxyVal = secondaryList[i].getProxyValue(proxyPos);
        float varVal;
        if (currentFrequency == daily)
        {
            varVal = secondaryList[i].getMeteoPointValueD(getCrit3DDate(currentDateTime.date()), myVar);
        }
        else if (currentFrequency == hourly)
        {
            varVal = secondaryList[i].getMeteoPointValueH(getCrit3DDate(currentDateTime.date()), currentDateTime.time().hour(), currentDateTime.time().minute(), myVar);
        }
        if (proxyVal != NODATA && varVal != NODATA)
        {
            point.setX(proxyVal);
            point.setY(varVal);
            point_vector2.append(point);
        }

    }
    for (int i = 0; i<supplementalList.size(); i++)
    {
        float proxyVal = supplementalList[i].getProxyValue(proxyPos);
        float varVal;
        if (currentFrequency == daily)
        {
            varVal = supplementalList[i].getMeteoPointValueD(getCrit3DDate(currentDateTime.date()), myVar);
        }
        else if (currentFrequency == hourly)
        {
            varVal = supplementalList[i].getMeteoPointValueH(getCrit3DDate(currentDateTime.date()), currentDateTime.time().hour(), currentDateTime.time().minute(), myVar);
        }
        if (proxyVal != NODATA && varVal != NODATA)
        {
            point.setX(proxyVal);
            point.setY(varVal);
            point_vector3.append(point);
        }

    }
    chartView->drawPointSeriesPrimary(point_vector);
    chartView->drawPointSeriesSecondary(point_vector2);
    chartView->drawPointSeriesSupplemental(point_vector3);
}

void Crit3DProxyWidget::climatologicalLRClicked(int toggled)
{
    chartView->cleanClimLapseRate();
    if (toggled)
    {
        chartView->drawClimLapseRate();
    }
}

void Crit3DProxyWidget::computeHighestStationIndex()
{
    double zMaxPrimary = 0;
    double zMaxSecondary = 0;
    double zMaxSupplemental = 0;

    int highestStationIndexPrimary = 0;
    int highestStationIndexSecondary = 0;
    int highestStationIndexSupplemental = 0;

    for (int i = 0; i<primaryList.size(); i++)
    {
        if (primaryList[i].point.z > zMaxPrimary)
        {
            highestStationIndexPrimary = i;
            zMaxPrimary = primaryList[i].point.z;
        }
    }

    for (int i = 0; i<secondaryList.size(); i++)
    {
        if (secondaryList[i].point.z > zMaxSecondary)
        {
            highestStationIndexSecondary = i;
            zMaxSecondary = secondaryList[i].point.z;
        }
    }

    for (int i = 0; i<supplementalList.size(); i++)
    {
        if (supplementalList[i].point.z > zMaxSupplemental)
        {
            highestStationIndexSupplemental = i;
            zMaxSupplemental = supplementalList[i].point.z;
        }
    }

    if (std::max(zMaxPrimary, zMaxSecondary) == zMaxPrimary)
    {
        if (std::max(zMaxPrimary, zMaxSupplemental) == zMaxPrimary)
        {
            highestStationBelongToList = 0;
            highestStationIndex = highestStationIndexPrimary;
            zMax = zMaxPrimary;
        }
        else
        {
            highestStationBelongToList = 2;
            highestStationIndex = highestStationIndexSupplemental;
            zMax = zMaxSupplemental;
        }
    }
    else
    {
        if (std::max(zMaxSecondary, zMaxSupplemental) == zMaxSecondary)
        {
            highestStationBelongToList = 1;
            highestStationIndex = highestStationIndexSecondary;
            zMax = zMaxSecondary;
        }
        else
        {
            highestStationBelongToList = 2;
            highestStationIndex = highestStationIndexSupplemental;
            zMax = zMaxSupplemental;
        }
    }
}

