#include "dialogMeteoTable.h"
#include "utilities.h"
#include "commonConstants.h"

DialogMeteoTable::DialogMeteoTable(Crit3DMeteoSettings *meteoSettings_, QVector<Crit3DMeteoPoint> meteoPoints, QDate firstDate, QDate lastDate, frequencyType currentFreq, QList<QString> currentVariables)
    :meteoPoints(meteoPoints), firstDate(firstDate), lastDate(lastDate), currentFreq(currentFreq), currentVariables(currentVariables)
{
    meteoSettings = meteoSettings_;

    QString title = "Table meteo values ID: ";
    QList<QString> idList;
    QList<QString> nameList;
    for (int i=0; i<meteoPoints.size(); i++)
    {
        idList << QString::fromStdString(meteoPoints[i].id);

        QString pointName = QString::fromStdString(meteoPoints[i].name);
        QList<QString> elementsName = pointName.split(' ');
        if (elementsName.size() == 1)
        {
            pointName = elementsName[0].left(8);
        }
        else
        {
            pointName = elementsName[0].left(4)+elementsName[elementsName.size()-1].left(4);
        }
        nameList << pointName;
    }
    title = title+idList.join(",");
    this->setWindowTitle(title);
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(800, 600);


    meteoTable = new MeteoTable();
    mainLayout->addWidget(meteoTable);

    int colNumber = currentVariables.size()*meteoPoints.size()+1; // variables for ID + Data
    int rowNumber = 0;

    if (currentFreq == daily)
    {
        rowNumber = firstDate.daysTo(lastDate)+1; // naVlues = nDays
    }
    else if (currentFreq == hourly)
    {
        rowNumber = (firstDate.daysTo(lastDate)+1)*24; // naVlues = nDays * 24 hours
    }

    meteoTable->setRowCount(rowNumber);
    meteoTable->setColumnCount(colNumber);

    std::string nameField;

    labels.clear();
    meteoTableHeader.clear();
    meteoTableHeader << "Date";

    for (int i=0; i < currentVariables.size(); i++)
    {
        for (int mp = 0; mp < meteoPoints.size(); mp++)
        {
            meteoTableHeader << idList[mp]+"_"+nameList[mp]+"_"+currentVariables[i];
        }
    }

    for (int j = 1; j < rowNumber-1; j++)
    {
        labels << QString::number(j);
    }

    QDate myDate;
    QDateTime firstDateTime(firstDate, QTime(0,0,0), Qt::UTC);
    QDateTime myDateTime;

    for (int row=0; row < rowNumber; row++)
    {
        for (int col = 0; col < colNumber; col++)
        {
            if (col == 0)
            {
                if (currentFreq == daily)
                {
                    myDate = firstDate.addDays(row);
                    meteoTable->setItem(row, col, new QTableWidgetItem(myDate.toString("yyyy-MM-dd")));
                }
                else if (currentFreq == hourly)
                {
                    myDateTime = firstDateTime.addSecs(row * 3600);
                    meteoTable->setItem(row, col, new QTableWidgetItem(myDateTime.toString("yyyy-MM-dd hh:mm")));
                }
            }
            else
            {
                int varPos = (col-1)/meteoPoints.size();
                int mpPos = (col % meteoPoints.size());
                if (mpPos != 0)
                {
                    mpPos = mpPos - 1;
                }
                else
                {
                    mpPos = meteoPoints.size()-1;
                }
                if (currentFreq == daily)
                {
                    meteoVariable meteoVar = MapDailyMeteoVar.at(currentVariables[varPos].toStdString());
                    double value = meteoPoints[mpPos].getMeteoPointValueD(getCrit3DDate(myDate), meteoVar, meteoSettings);
                    if (value != NODATA)
                    {
                        meteoTable->setItem(row, col, new QTableWidgetItem( QString::number(value)));
                    }
                }
                else if (currentFreq == hourly)
                {
                    meteoVariable meteoVar = MapHourlyMeteoVar.at(currentVariables[varPos].toStdString());
                    double value = meteoPoints[mpPos].getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, meteoVar);
                    if (value != NODATA)
                    {
                        meteoTable->setItem(row, col, new QTableWidgetItem( QString::number(value)));
                    }
                }

            }
        }
    }


    meteoTable->setVerticalHeaderLabels(labels);
    meteoTable->setHorizontalHeaderLabels(meteoTableHeader);
    meteoTable->resizeColumnsToContents();
    meteoTable->setSelectionBehavior(QAbstractItemView::SelectItems);
    meteoTable->setSelectionMode(QAbstractItemView::ContiguousSelection);
    meteoTable->setShowGrid(true);
    meteoTable->horizontalHeader()->setStyleSheet("QHeaderView { font-weight: bold; }");
    meteoTable->setStyleSheet("QTableView {selection-background-color: red;}");


    setLayout(mainLayout);
    exec();
}

DialogMeteoTable::~DialogMeteoTable()
{
    close();
}
