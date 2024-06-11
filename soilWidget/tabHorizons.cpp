#include "tabHorizons.h"
#include "commonConstants.h"
#include "soil.h"


TabHorizons::TabHorizons()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QLabel* dbTableLabel = new QLabel("Soil parameters from DB:");
    dbTableLabel->setStyleSheet("font: 11pt;");
    QLabel* modelTableLabel = new QLabel("Soil parameters estimated by model:");
    modelTableLabel->setStyleSheet("font: 11pt;");
    QVBoxLayout *tableLayout = new QVBoxLayout;
    tableDb = new Crit3DSoilTable(dbTable);
    tableModel = new Crit3DSoilTable(modelTable);
    QHBoxLayout *addDeleteRowLayout = new QHBoxLayout;
    QLabel* addDeleteLabel = new QLabel("Modify horizons:");
    addRow = new QPushButton("+");
    addRow->setFixedWidth(40);
    deleteRow = new QPushButton("-");
    deleteRow->setFixedWidth(40);
    addRow->setEnabled(false);
    deleteRow->setEnabled(false);
    addDeleteRowLayout->addStretch(40);
    addDeleteRowLayout->addWidget(addDeleteLabel);
    addDeleteRowLayout->addWidget(addRow);
    addDeleteRowLayout->addWidget(deleteRow);

    // tableModel with a Width perfectly wrapping all columns, tableDb copy the same size, add margin because now the table is empty
    tableDb->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tableModel->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    int margin = 25;
    tableModel->setFixedWidth(tableModel->horizontalHeader()->length() + tableModel->verticalHeader()->width() + margin);
    tableDb->setFixedWidth(tableModel->width());

    connect(tableDb->verticalHeader(), &QHeaderView::sectionClicked, [=](int index){ this->tableDbVerticalHeaderClick(index); });
    connect(tableModel->verticalHeader(), &QHeaderView::sectionClicked, [=](int index){ this->tableDbVerticalHeaderClick(index); });
    connect(addRow, &QPushButton::clicked, [=](){ this->addRowClicked(); });
    connect(deleteRow, &QPushButton::clicked, [=](){ this->removeRowClicked(); });

    tableLayout->addWidget(dbTableLabel);
    tableLayout->addWidget(tableDb);
    tableLayout->addLayout(addDeleteRowLayout);
    tableLayout->addWidget(modelTableLabel);
    tableLayout->addWidget(tableModel);

    mainLayout->addWidget(barHorizons.groupBox);
    mainLayout->addLayout(tableLayout);

    setLayout(mainLayout);
    insertSoilElement = false;
}


void TabHorizons::updateBarHorizon(soil::Crit3DSoil* mySoil)
{
    if (mySoil == nullptr)
    {
        return;
    }

    barHorizons.clear();
    barHorizons.draw(mySoil);
}


void TabHorizons::insertSoilHorizons(soil::Crit3DSoil *soil, std::vector<soil::Crit3DTextureClass> *textureClassList,
                                     std::vector<soil::Crit3DGeotechnicsClass> *geotechnicsClassList,
                                     soil::Crit3DFittingOptions *fittingOptions)
{
    if (soil == nullptr)
    {
        return;
    }
    resetAll();

    barHorizons.draw(soil);

    insertSoilElement = true;
    //disable events otherwise setBackground call again cellChanged event
    tableDb->blockSignals(true);
    mySoil = soil;
    myTextureClassList = textureClassList;
    myGeotechnicsClassList = geotechnicsClassList;
    myFittingOptions = fittingOptions;

    int row = signed(mySoil->nrHorizons);
    tableDb->setRowCount(row);
    tableModel->setRowCount(row);

    // fill Tables
    for (int i = 0; i < row; i++)
    {
        tableDb->setItem(i, 0, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.upperDepth, 'f', 0)));
        tableDb->item(i,0)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 1, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.lowerDepth, 'f', 0)));
        tableDb->item(i,1)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 2, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.sand, 'f', 1 )));
        tableDb->item(i,2)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 3, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.silt, 'f', 1 )));
        tableDb->item(i,3)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 4, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.clay , 'f', 1)));
        tableDb->item(i,4)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 5, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.coarseFragments, 'f', 1 )));
        tableDb->item(i,5)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 6, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.organicMatter, 'f', 1 )));
        tableDb->item(i,6)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 7, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.bulkDensity, 'f', 3 )));
        tableDb->item(i,7)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 8, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.kSat, 'f', 3 )));
        tableDb->item(i,8)->setTextAlignment(Qt::AlignRight);
        tableDb->setItem(i, 9, new QTableWidgetItem( QString::number(mySoil->horizon[i].dbData.thetaSat, 'f', 3)));
        tableDb->item(i,9)->setTextAlignment(Qt::AlignRight);

        tableModel->setItem(i, 0, new QTableWidgetItem( QString::fromStdString(mySoil->horizon[i].texture.classNameUSDA)));
        if (mySoil->horizon[i].coarseFragments != NODATA)
        {
            tableModel->setItem(i, 1, new QTableWidgetItem( QString::number(mySoil->horizon[i].coarseFragments*100, 'f', 1 )));
        }
        else
        {
            tableModel->setItem(i, 1, new QTableWidgetItem( QString::number(mySoil->horizon[i].coarseFragments, 'f', 1 )));
        }
        tableModel->item(i,1)->setTextAlignment(Qt::AlignRight);

        if (mySoil->horizon[i].organicMatter != NODATA)
        {
            tableModel->setItem(i, 2, new QTableWidgetItem( QString::number(mySoil->horizon[i].organicMatter*100, 'f', 1 )));
        }
        else
        {
            tableModel->setItem(i, 2, new QTableWidgetItem( QString::number(mySoil->horizon[i].organicMatter, 'f', 1 )));
        }
        tableModel->item(i,2)->setTextAlignment(Qt::AlignRight);

        tableModel->setItem(i, 3, new QTableWidgetItem( QString::number(mySoil->horizon[i].bulkDensity, 'f', 3 )));
        tableModel->item(i,3)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 4, new QTableWidgetItem( QString::number(mySoil->horizon[i].waterConductivity.kSat, 'f', 3 )));
        tableModel->item(i,4)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 5, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.thetaS, 'f', 3 )));
        tableModel->item(i,5)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 6, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.thetaR, 'f', 3 )));
        tableModel->item(i,6)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 7, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.he, 'f', 3 )));
        tableModel->item(i,7)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 8, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.alpha, 'f', 3 )));
        tableModel->item(i,8)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 9, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.n, 'f', 3 )));
        tableModel->item(i,9)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 10, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.m, 'f', 3 )));   
        tableModel->item(i,10)->setTextAlignment(Qt::AlignRight);
    }

    // check all Depths
    checkDepths();
    // check other values
    for (int i = 0; i < row; i++)
    {
        checkMissingItem(i);
        if (checkHorizonData(i))
        {
            checkComputedValues(i);
        }
    }

    tableDb->blockSignals(false);
    addRow->setEnabled(true);
    deleteRow->setEnabled(false);

    connect(tableDb, &QTableWidget::cellChanged, [=](int row, int column){ this->cellChanged(row, column); });
    connect(tableDb, &QTableWidget::cellClicked, [=](int row, int column){ this->cellClickedDb(row, column); });
    connect(tableModel, &QTableWidget::cellClicked, [=](int row, int column){ this->cellClickedModel(row, column); });

    for (int i=0; i < barHorizons.barList.size(); i++)
    {
        connect(barHorizons.barList[i], SIGNAL(clicked(int)), this, SLOT(widgetClicked(int)));
    }

    cellClickedDb(0,0);

}

void TabHorizons::updateTableModel(soil::Crit3DSoil *soil)
{
    if (soil == nullptr)
    {
        return;
    }

    // reset tableModel
    tableModel->setRowCount(0);
    insertSoilElement = false;
    clearSelections();

    mySoil = soil;

    int row = signed(mySoil->nrHorizons);
    tableModel->setRowCount(row);

    // fill Tables
    for (int i = 0; i < row; i++)
    {
        tableModel->setItem(i, 0, new QTableWidgetItem( QString::fromStdString(mySoil->horizon[i].texture.classNameUSDA)));
        if (mySoil->horizon[i].coarseFragments != NODATA)
        {
            tableModel->setItem(i, 1, new QTableWidgetItem( QString::number(mySoil->horizon[i].coarseFragments*100, 'f', 1 )));
        }
        else
        {
            tableModel->setItem(i, 1, new QTableWidgetItem( QString::number(mySoil->horizon[i].coarseFragments, 'f', 1 )));
        }
        tableModel->item(i,1)->setTextAlignment(Qt::AlignRight);

        if (mySoil->horizon[i].organicMatter != NODATA)
        {
            tableModel->setItem(i, 2, new QTableWidgetItem( QString::number(mySoil->horizon[i].organicMatter*100, 'f', 1 )));
        }
        else
        {
            tableModel->setItem(i, 2, new QTableWidgetItem( QString::number(mySoil->horizon[i].organicMatter, 'f', 1 )));
        }
        tableModel->item(i,2)->setTextAlignment(Qt::AlignRight);

        tableModel->setItem(i, 3, new QTableWidgetItem( QString::number(mySoil->horizon[i].bulkDensity, 'f', 3 )));
        tableModel->item(i,3)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 4, new QTableWidgetItem( QString::number(mySoil->horizon[i].waterConductivity.kSat, 'f', 3 )));
        tableModel->item(i,4)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 5, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.thetaS, 'f', 3 )));
        tableModel->item(i,5)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 6, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.thetaR, 'f', 3 )));
        tableModel->item(i,6)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 7, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.he, 'f', 3 )));
        tableModel->item(i,7)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 8, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.alpha, 'f', 3 )));
        tableModel->item(i,8)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 9, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.n, 'f', 3 )));
        tableModel->item(i,9)->setTextAlignment(Qt::AlignRight);
        tableModel->setItem(i, 10, new QTableWidgetItem( QString::number(mySoil->horizon[i].vanGenuchten.m, 'f', 3 )));
        tableModel->item(i,10)->setTextAlignment(Qt::AlignRight);
    }

    // check other values
    for (int i = 0; i < row; i++)
    {
        checkMissingItem(i);
        if (checkHorizonData(i))
        {
            checkComputedValues(i);
        }
    }
}


bool TabHorizons::checkDepths()
{
    bool depthsOk = true;
    // reset background color
    for (int horizonNum = 0; horizonNum < tableDb->rowCount(); horizonNum++)
    {
        tableDb->item(horizonNum,0)->setBackground(Qt::white);
        tableDb->item(horizonNum,1)->setBackground(Qt::white);
    }
    for (int horizonNum = 0; horizonNum<tableDb->rowCount(); horizonNum++)
    {
        //except first row
        if ( horizonNum > 0)
        {
            if (mySoil->horizon[unsigned(horizonNum)].dbData.upperDepth != mySoil->horizon[horizonNum-1].dbData.lowerDepth)
            {
                tableDb->item(horizonNum,0)->setBackground(Qt::red);
                tableDb->item(horizonNum-1,1)->setBackground(Qt::red);
                depthsOk = false;
            }
        }

        // except last row
        if (horizonNum < tableDb->rowCount()-1)
        {
            if (mySoil->horizon[unsigned(horizonNum)].dbData.lowerDepth != mySoil->horizon[horizonNum+1].dbData.upperDepth)
            {
                tableDb->item(horizonNum,1)->setBackground(Qt::red);
                tableDb->item(horizonNum+1,0)->setBackground(Qt::red);
                depthsOk = false;
            }
        }

        if (mySoil->horizon[unsigned(horizonNum)].dbData.upperDepth == NODATA || mySoil->horizon[horizonNum].dbData.lowerDepth == NODATA)
        {
            tableDb->item(horizonNum,0)->setBackground(Qt::red);
            tableDb->item(horizonNum,1)->setBackground(Qt::red);
            depthsOk = false;
        }
        else if (mySoil->horizon[unsigned(horizonNum)].dbData.upperDepth > mySoil->horizon[horizonNum].dbData.lowerDepth)
        {
            tableDb->item(horizonNum,0)->setBackground(Qt::red);
            tableDb->item(horizonNum,1)->setBackground(Qt::red);
            depthsOk = false;
        }
        else
        {
            if (mySoil->horizon[unsigned(horizonNum)].dbData.upperDepth < 0)
            {
                tableDb->item(horizonNum,0)->setBackground(Qt::red);
                depthsOk = false;
            }
            if (mySoil->horizon[unsigned(horizonNum)].dbData.lowerDepth < 0)
            {
                tableDb->item(horizonNum,1)->setBackground(Qt::red);
                depthsOk = false;
            }
        }
    }
    return depthsOk;
}


bool TabHorizons::checkHorizonData(int horizonNum)
{
    bool goOn = true;
    soil::Crit3DHorizonDbData* dbData = &(mySoil->horizon[unsigned(horizonNum)].dbData);

    if (soil::getUSDATextureClass(dbData->sand, dbData->silt, dbData->clay) == NODATA)
    {
        tableDb->item(horizonNum,2)->setBackground(Qt::red);
        tableDb->item(horizonNum,3)->setBackground(Qt::red);
        tableDb->item(horizonNum,4)->setBackground(Qt::red);

        setInvalidTableModelRow(horizonNum);
        goOn = false;
    }

    if (dbData->coarseFragments != NODATA && (dbData->coarseFragments < 0 || dbData->coarseFragments >= 100))
    {
        tableDb->item(horizonNum,5)->setBackground(Qt::red);
    }

    if ( dbData->organicMatter != NODATA && ((dbData->organicMatter < 0) || (dbData->organicMatter > 100)) )
    {
        tableDb->item(horizonNum,6)->setBackground(Qt::red);
    }

    if (dbData->bulkDensity != NODATA && (dbData->bulkDensity <= 0 || dbData->bulkDensity > QUARTZ_DENSITY))
    {
        tableDb->item(horizonNum,7)->setBackground(Qt::red);
    }

    if (dbData->kSat != NODATA && dbData->kSat <= 0)
    {
        tableDb->item(horizonNum,8)->setBackground(Qt::red);
    }

    if (dbData->thetaSat != NODATA && (dbData->thetaSat <= 0 || dbData->thetaSat >= 1))
    {
        tableDb->item(horizonNum,9)->setBackground(Qt::red);
    }

    return goOn;
}


void TabHorizons::setInvalidTableModelRow(int horizonNum)
{
    tableModel->item(horizonNum,0)->setText("UNDEFINED");
    tableModel->item(horizonNum,0)->setBackground(Qt::red);
    tableModel->item(horizonNum,1)->setBackground(Qt::red);
    tableModel->item(horizonNum,2)->setBackground(Qt::red);
    tableModel->item(horizonNum,3)->setBackground(Qt::red);
    tableModel->item(horizonNum,4)->setBackground(Qt::red);
    tableModel->item(horizonNum,5)->setBackground(Qt::red);
    tableModel->item(horizonNum,6)->setBackground(Qt::red);
    tableModel->item(horizonNum,7)->setBackground(Qt::red);
    tableModel->item(horizonNum,8)->setBackground(Qt::red);
    tableModel->item(horizonNum,9)->setBackground(Qt::red);
    tableModel->item(horizonNum,10)->setBackground(Qt::red);
}


void TabHorizons::checkMissingItem(int horizonNr)
{
    QString NODATAString = "-9999";

    int nrColumns = tableDb->columnCount();
    // except for lower and upper depths (columns 0 and 1)
    for (int j = 2; j < nrColumns; j++)
    {
        if (tableDb->item(horizonNr, j) != nullptr)
        {
            if (tableDb->item(horizonNr, j)->text().contains(NODATAString) || tableDb->item(horizonNr,j )->text().isEmpty())
            {
                tableDb->item(horizonNr, j)->setBackground(Qt::yellow);
                tableDb->item(horizonNr, j)->setText("");
            }
        }
    }

    nrColumns = tableModel->columnCount();
    for (int j = 0; j < nrColumns; j++)
    {
        if (tableModel->item(horizonNr, j) != nullptr)
        {
            if (tableModel->item(horizonNr, j)->text().contains(NODATAString) || tableModel->item(horizonNr, j)->text().isEmpty())
            {
                tableModel->item(horizonNr, j)->setBackground(Qt::red);
                tableModel->item(horizonNr, j)->setText("");
            }
        }
    }
}


void TabHorizons::checkComputedValues(int horizonNum)
{
    soil::Crit3DHorizon * horizon = &(mySoil->horizon[unsigned(horizonNum)]);

    if (abs(horizon->dbData.coarseFragments - horizon->coarseFragments*100) > EPSILON)
    {
        tableModel->item(horizonNum,1)->setBackground(Qt::yellow);
    }

    if (abs(horizon->dbData.organicMatter - horizon->organicMatter*100) > EPSILON)
    {
        tableModel->item(horizonNum,2)->setBackground(Qt::yellow);
    }

    if (abs(horizon->dbData.bulkDensity - horizon->bulkDensity) > EPSILON)
    {
        tableModel->item(horizonNum,3)->setBackground(Qt::yellow);
    }

    if (abs(horizon->dbData.kSat - horizon->waterConductivity.kSat) > EPSILON)
    {
        tableModel->item(horizonNum,4)->setBackground(Qt::yellow);
    }

    if (abs(horizon->dbData.thetaSat - horizon->vanGenuchten.thetaS) > EPSILON)
    {
        tableModel->item(horizonNum,5)->setBackground(Qt::yellow);
    }
}


void TabHorizons::clearSelections()
{
    tableDb->clearSelection();
    tableModel->clearSelection();
    deleteRow->setEnabled(false);
    emit horizonSelected(-1);
}

void TabHorizons::cellClickedDb(int row, int column)
{
    clearSelections();
    tableDb->setSelectionBehavior(QAbstractItemView::SelectItems);
    tableModel->setSelectionBehavior(QAbstractItemView::SelectItems);
    tableDb->item(row, column)->setSelected(true);

    switch (column) {
        case 5:
        {
            tableModel->item(row,1)->setSelected(true);
            break;
        }
        case 6:
        {
            tableModel->item(row,2)->setSelected(true);
            break;
        }
        case 7:
        {
            tableModel->item(row,3)->setSelected(true);
            break;
        }
        case 8:
        {
            tableModel->item(row,4)->setSelected(true);
            break;
        }
        case 9:
        {
            tableModel->item(row,5)->setSelected(true);
            break;
        }
    }

    barHorizons.selectItem(row);

    deleteRow->setEnabled(false);
    emit horizonSelected(row);
}

void TabHorizons::cellClickedModel(int row, int column)
{

    clearSelections();
    tableDb->setSelectionBehavior(QAbstractItemView::SelectItems);
    tableModel->setSelectionBehavior(QAbstractItemView::SelectItems);
    tableModel->item(row, column)->setSelected(true);

    switch (column) {
        case 1:
        {
            tableDb->item(row, 5)->setSelected(true);
            break;
        }
        case 2:
        {
            tableDb->item(row, 6)->setSelected(true);
            break;
        }
        case 3:
        {
            tableDb->item(row, 7)->setSelected(true);
            break;
        }
        case 4:
        {
            tableDb->item(row, 8)->setSelected(true);
            break;
        }
        case 5:
        {
            tableDb->item(row, 9)->setSelected(true);
            break;
        }
    }

    barHorizons.selectItem(row);

    deleteRow->setEnabled(false);
    emit horizonSelected(row);
}

void TabHorizons::tableDbVerticalHeaderClick(int index)
{
    tableDb->setSelectionBehavior(QAbstractItemView::SelectRows);
    tableModel->setSelectionBehavior(QAbstractItemView::SelectRows);

    tableDb->selectRow(index);
    tableDb->horizontalHeader()->setHighlightSections(false);
    tableModel->selectRow(index);
    tableModel->horizontalHeader()->setHighlightSections(false);
    deleteRow->setEnabled(true);

    barHorizons.selectItem(index);

    emit horizonSelected(index);

}

void TabHorizons::cellChanged(int row, int column)
{
    if (tableDb->itemAt(row,column) == nullptr || mySoil->nrHorizons < unsigned(row))
    {
        qDebug() << "mySoil->horizon->dbData.horizonNr < row ";
        return;
    }

    //disable events otherwise setBackground call again cellChanged event
    tableDb->blockSignals(true);
    tableModel->selectRow(row);
    QString data = tableDb->item(row, column)->text();
    data.replace(",", ".");

    // set new value
    switch (column) {
        case 0:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.upperDepth = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.upperDepth = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 0));
            }
            break;
        }
        case 1:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.lowerDepth = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.lowerDepth = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 0));
            }
            break;
        }
        case 2:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.sand = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.sand = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            break;
        }
        case 3:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.silt = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.silt = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            break;
        }
        case 4:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.clay = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.clay = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            break;
        }
        case 5:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.coarseFragments = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.coarseFragments = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            break;
        }
        case 6:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.organicMatter = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.organicMatter = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            break;
        }
        case 7:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.bulkDensity = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.bulkDensity = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 3));
            }
            break;
        }
        case 8:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.kSat = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.kSat = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 3));
            }
            break;
        }
        case 9:
        {
            if (data == QString::number(NODATA) || data.isEmpty())
            {
                mySoil->horizon[unsigned(row)].dbData.thetaSat = NODATA;
                tableDb->item(row, column)->setText("");
            }
            else
            {
                mySoil->horizon[unsigned(row)].dbData.thetaSat = data.toDouble();
                tableDb->item(row, column)->setText(QString::number(data.toDouble(), 'f', 3));
            }
            break;
        }
    }

    std::string errorString;
    soil::setHorizon(mySoil->horizon[unsigned(row)], *myTextureClassList, *myGeotechnicsClassList, *myFittingOptions, errorString);

    // update tableModel values
    tableModel->item(row,0)->setText(QString::fromStdString(mySoil->horizon[unsigned(row)].texture.classNameUSDA));

    if (mySoil->horizon[unsigned(row)].coarseFragments != NODATA)
    {
        tableModel->item(row,1)->setText(QString::number(mySoil->horizon[unsigned(row)].coarseFragments*100, 'f', 1 ));
    }
    else
    {
        tableModel->item(row,1)->setText("");
    }

    if (mySoil->horizon[unsigned(row)].organicMatter != NODATA)
    {
        tableModel->item(row,2)->setText(QString::number(mySoil->horizon[unsigned(row)].organicMatter*100, 'f', 1 ));
    }
    else
    {
        tableModel->item(row,2)->setText("");
    }

    tableModel->item(row,3)->setText(QString::number(mySoil->horizon[unsigned(row)].bulkDensity, 'f', 3));
    tableModel->item(row,4)->setText(QString::number(mySoil->horizon[unsigned(row)].waterConductivity.kSat, 'f', 3));
    tableModel->item(row,5)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.thetaS, 'f', 3));
    tableModel->item(row,6)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.thetaR, 'f', 3));
    tableModel->item(row,7)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.he, 'f', 3));
    tableModel->item(row,8)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.alpha, 'f', 3));
    tableModel->item(row,9)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.n, 'f', 3));
    tableModel->item(row,10)->setText(QString::number(mySoil->horizon[unsigned(row)].vanGenuchten.m, 'f', 3));

    // reset background color for the row changed
    for (int j = 0; j < tableDb->columnCount(); j++)
    {
        tableDb->item(row,j)->setBackground(Qt::white);
    }

    for (int j = 0; j < tableModel->columnCount(); j++)
    {
        tableModel->item(row,j)->setBackground(Qt::white);
    }

    // check all Depths
    bool depthsOk = checkDepths();

    // check new values and assign background color
    checkMissingItem(row);
    bool checkHorizon = checkHorizonData(row);
    if (checkHorizon)
    {
        checkComputedValues(row);
    }

    updateBarHorizon(mySoil);
    tableDb->blockSignals(false);

    if (depthsOk && checkHorizon)
    {
        emit horizonSelected(row);
        emit updateSignal();
    }
}


void TabHorizons::addRowClicked()
{
    tableDb->blockSignals(true);
    int numRow;

    if (tableDb->rowCount() != 0)
    {
        if (tableDb->selectedItems().isEmpty())
        {
            QMessageBox::critical(nullptr, "Warning", "Select the row of the horizon before the one you want to add");
            return;
        }
        else
        {
            if (tableDb->selectedItems().size() != tableDb->columnCount())
            {
                QMessageBox::critical(nullptr, "Warning", "Select the row of the horizon before the one you want to add");
                return;
            }
            numRow = tableDb->selectedItems().at(0)->row()+1;
        }
    }
    else
    {
        numRow = 0;
    }

    tableDb->insertRow(numRow);
    tableModel->insertRow(numRow);

    for (int j=0; j < tableDb->columnCount(); j++)
    {
        tableDb->setItem(numRow, j, new QTableWidgetItem());
        tableDb->item(numRow,j)->setTextAlignment(Qt::AlignRight);
    }

    for (int j=0; j < tableModel->columnCount(); j++)
    {
        tableModel->setItem(numRow, j, new QTableWidgetItem());
        if (j>0)
        {
            tableModel->item(numRow,j)->setTextAlignment(Qt::AlignRight);
        }
    }
    deleteRow->setEnabled(true);

    setInvalidTableModelRow(numRow);

    soil::Crit3DHorizon* newHorizon = new soil::Crit3DHorizon();
    // set newHorizon dbData
    newHorizon->dbData.horizonNr = numRow;
    QString lowerDepth;
    if (numRow != 0)
    {
        lowerDepth = tableDb->item(numRow-1, 1)->text();
        tableDb->item(numRow, 0)->setText(lowerDepth);
        newHorizon->dbData.upperDepth = lowerDepth.toDouble();
    }
    else
    {
        newHorizon->dbData.upperDepth = 0;
        tableDb->item(numRow, 0)->setText("0");
    }
    newHorizon->dbData.lowerDepth = NODATA;
    newHorizon->dbData.sand = NODATA;
    newHorizon->dbData.silt = NODATA;
    newHorizon->dbData.clay = NODATA;
    newHorizon->dbData.coarseFragments = NODATA;
    newHorizon->dbData.organicMatter = NODATA;
    newHorizon->dbData.bulkDensity = NODATA;
    newHorizon->dbData.thetaSat = NODATA;
    newHorizon->dbData.kSat = NODATA;

    mySoil->addHorizon(numRow, *newHorizon);
    // check all Depths
    checkDepths();
    // check new values and assign background color
    checkMissingItem(numRow);
    if (checkHorizonData(numRow))
    {
        checkComputedValues(numRow);
    }
    tableDb->blockSignals(false);

}

void TabHorizons::removeRowClicked()
{
    tableDb->blockSignals(true);
    int row;
    // check if a row is selected
    if (tableDb->selectedItems().isEmpty())
    {
        QMessageBox::critical(nullptr, "Error!", "Select a horizon");
        return;
    }
    if (tableDb->selectedItems().size() == tableDb->columnCount())
    {
        row = tableDb->selectedItems().at(0)->row();
    }
    else
    {
        QMessageBox::critical(nullptr, "Error!", "Select a horizon");
        return;
    }
    tableDb->removeRow(row);
    tableModel->removeRow(row);
    mySoil->deleteHorizon(row);
    // check all Depths
    bool depthsOk = checkDepths();
    tableDb->blockSignals(false);

    if (depthsOk)
    {
        emit updateSignal();
    }

}

void TabHorizons::resetAll()
{
    // delete all Widgets
    barHorizons.clear();
    tableDb->setRowCount(0);
    tableModel->setRowCount(0);
    insertSoilElement = false;
    clearSelections();
}

bool TabHorizons::getInsertSoilElement() const
{
    return insertSoilElement;
}

void TabHorizons::setInsertSoilElement(bool value)
{
    insertSoilElement = value;
}

void TabHorizons::widgetClicked(int index)
{
    // check selection state
    if (barHorizons.barList[index]->getSelected())
    {
        tableDbVerticalHeaderClick(index);
    }
    else
    {
        clearSelections();
    }

}

void TabHorizons::copyTableDb()
{
    tableDb->copyAll();
}

void TabHorizons::copyTableModel()
{
    tableModel->copyAll();
}

void TabHorizons::exportTableDb(QString csvFile)
{
    tableDb->exportToCsv(csvFile);
}

void TabHorizons::exportTableModel(QString csvFile)
{
    tableModel->exportToCsv(csvFile);
}


