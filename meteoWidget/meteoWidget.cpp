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
    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.
    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/


#include "meteoWidget.h"
#include "utilities.h"

#include <QMessageBox>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QDate>

#include <QDebug>


Crit3DMeteoWidget::Crit3DMeteoWidget()
{
    this->setWindowTitle(QStringLiteral("Graph"));
    this->resize(1240, 700);

    QString csvPath, defaultPath, stylesPath;
    if (searchDataPath(&csvPath))
    {
        defaultPath = csvPath + "SETTINGS/Crit3DPlotDefault.csv";
        stylesPath = csvPath + "SETTINGS/Crit3DPlotStyles.csv";
    }

    // read Crit3DPlotDefault and fill MapCSVDefault
    int CSVRequiredInfo = 3;
    QFile fileDefaultGraph(defaultPath);
    if ( !fileDefaultGraph.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
    }
    else
    {
        QTextStream in(&fileDefaultGraph);
        in.readLine(); //skip first line
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < CSVRequiredInfo)
            {
                qDebug() << "invalid format CSV, missing data";
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                qDebug() << "invalid format CSV, missing data";
            }
            MapCSVDefault.insert(key,items);
        }
    }

    // read Crit3DPlotStyles and fill MapCSVStyles
    QFile fileStylesGraph(stylesPath);
    if ( !fileStylesGraph.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
    }
    else
    {
        QTextStream in(&fileStylesGraph);
        in.readLine(); //skip first line
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < CSVRequiredInfo)
            {
                qDebug() << "invalid format CSV, missing data";
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                qDebug() << "invalid format CSV, missing data";
            }
            MapCSVStyles.insert(key,items);
        }
    }

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    this->setLayout(mainLayout);
}

void Crit3DMeteoWidget::draw(Crit3DMeteoPoint* meteoPoint)
{

}



