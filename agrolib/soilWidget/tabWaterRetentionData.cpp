#include "tabWaterRetentionData.h"
#include "tableDelegateWaterRetention.h"
#include "tableWaterRetention.h"
#include "tableWidgetItem.h"
#include "commonConstants.h"
#include "soil.h"


TabWaterRetentionData::TabWaterRetentionData()
{
    QHBoxLayout* mainLayout = new QHBoxLayout;
    QVBoxLayout* tableLayout = new QVBoxLayout;
    tableWaterRetention = new TableWaterRetention();
    tableWaterRetention->setColumnCount(2);
    QList<QString> tableHeader;
    tableHeader << "Water potential [kPa]" << "Water content [m3 m-3]";
    tableWaterRetention->setHorizontalHeaderLabels(tableHeader);
    tableWaterRetention->setSelectionBehavior(QAbstractItemView::SelectRows);
    tableWaterRetention->resizeColumnsToContents();
    tableWaterRetention->setSelectionMode(QAbstractItemView::ContiguousSelection);
    tableWaterRetention->setStyleSheet("QTableView::item:selected { color:black;  border: 3px solid black}");
    tableWaterRetention->setShowGrid(true);
    tableWaterRetention->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    int margin = 50; // now table is empty
    tableWaterRetention->setFixedWidth(tableWaterRetention->horizontalHeader()->length() + tableWaterRetention->verticalHeader()->width() + margin);
    tableWaterRetention->setItemDelegate(new TableDelegateWaterRetention(tableWaterRetention));
    tableWaterRetention->setSortingEnabled(true);
    tableWaterRetention->sortByColumn(0, Qt::AscendingOrder);

    QHBoxLayout *addDeleteRowLayout = new QHBoxLayout;
    QLabel* addDeleteLabel = new QLabel("Modify data:");
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

    connect(tableWaterRetention->verticalHeader(), &QHeaderView::sectionClicked, [=](int index){ this->tableVerticalHeaderClick(index); });
    connect(addRow, &QPushButton::clicked, [=](){ this->addRowClicked(); });
    connect(deleteRow, &QPushButton::clicked, [=](){ this->removeRowClicked(); });

    tableLayout->addWidget(tableWaterRetention);
    tableLayout->addLayout(addDeleteRowLayout);
    mainLayout->addWidget(barHorizons.groupBox);
    mainLayout->addLayout(tableLayout);
    setLayout(mainLayout);

    fillData = false;
}

void TabWaterRetentionData::insertData(soil::Crit3DSoil *soil, soil::Crit3DTextureClass* textureClassList,
                                       soil::Crit3DFittingOptions* fittingOptions)
{
    if (soil == nullptr)
    {
        return;
    }

    resetAll();
    fillData = true;

    barHorizons.draw(soil);
    deleteRow->setEnabled(false);
    mySoil = soil;
    myTextureClassList = textureClassList;
    myFittingOptions = fittingOptions;
    addRow->setEnabled(true);

    connect(tableWaterRetention, &QTableWidget::cellClicked, [=](int row, int column){ this->cellClicked(row, column); });
    connect(tableWaterRetention, &QTableWidget::cellChanged, [=](int row, int column){ this->cellChanged(row, column); });

    for (int i=0; i < barHorizons.barList.size(); i++)
    {
        connect(barHorizons.barList[i], SIGNAL(clicked(int)), this, SLOT(widgetClicked(int)));
    }
    barHorizons.barList[0]->setSelected(true);
    widgetClicked(0);

}

void TabWaterRetentionData::tableVerticalHeaderClick(int index)
{
    tableWaterRetention->setSelectionBehavior(QAbstractItemView::SelectRows);
    tableWaterRetention->selectRow(index);
    tableWaterRetention->horizontalHeader()->setHighlightSections(false);
    deleteRow->setEnabled(true);
}

void TabWaterRetentionData::addRowClicked()
{
    tableWaterRetention->setSortingEnabled(false);
    tableWaterRetention->blockSignals(true);
    int numRow;
    if (tableWaterRetention->rowCount() != 0)
    {
        if (tableWaterRetention->selectedItems().isEmpty())
        {
            QMessageBox::critical(nullptr, "Warning", "Select the row of the horizon before the one you want to add");
            return;
        }
        else
        {
            if (tableWaterRetention->selectedItems().size() != tableWaterRetention->columnCount())
            {
                QMessageBox::critical(nullptr, "Warning", "Select the row of the horizon before the one you want to add");
                return;
            }
            numRow = tableWaterRetention->selectedItems().at(0)->row()+1;
        }
    }
    else
    {
        numRow = 0;
    }

    tableWaterRetention->insertRow(numRow);

    // fill default row (copy the previous row)
    if (numRow == 0)
    {
        tableWaterRetention->setItem(numRow, 0, new Crit3DTableWidgetItem(QString::number(0)));
        tableWaterRetention->setItem(numRow, 1, new Crit3DTableWidgetItem(QString::number(0)));
    }
    else
    {
        tableWaterRetention->setItem(numRow, 0, new Crit3DTableWidgetItem(tableWaterRetention->item(numRow-1,0)->text()));
        tableWaterRetention->setItem(numRow, 1, new Crit3DTableWidgetItem(tableWaterRetention->item(numRow-1,1)->text()));
    }
    tableWaterRetention->item(numRow,0)->setTextAlignment(Qt::AlignRight);
    tableWaterRetention->item(numRow,1)->setTextAlignment(Qt::AlignRight);

    tableWaterRetention->selectRow(numRow);

    soil::Crit3DWaterRetention newRow;
    newRow.water_potential = tableWaterRetention->item(numRow,0)->text().toDouble();
    newRow.water_content = tableWaterRetention->item(numRow,1)->text().toDouble();
    auto itPos = mySoil->horizon[currentHorizon].dbData.waterRetention.begin() + numRow;
    // Insert element
    mySoil->horizon[currentHorizon].dbData.waterRetention.insert(itPos, newRow);

    std::string errorString;
    soil::setHorizon(&(mySoil->horizon[currentHorizon]), myTextureClassList, myFittingOptions, errorString);

    deleteRow->setEnabled(true);

    if (!horizonChanged.contains(currentHorizon))
    {
        horizonChanged << currentHorizon;
    }

    tableWaterRetention->blockSignals(false);
    tableWaterRetention->setSortingEnabled(true);
    emit updateSignal();

}

void TabWaterRetentionData::removeRowClicked()
{
    int row;
    // check if a row is selected
    if (tableWaterRetention->selectedItems().isEmpty())
    {
        QMessageBox::critical(nullptr, "Error!", "Select a horizon");
        return;
    }
    if (tableWaterRetention->selectedItems().size() == tableWaterRetention->columnCount())
    {
        row = tableWaterRetention->selectedItems().at(0)->row();
    }
    else
    {
        QMessageBox::critical(nullptr, "Error!", "Select a horizon");
        return;
    }
    if (!horizonChanged.contains(currentHorizon))
    {
        horizonChanged << currentHorizon;
    }

    tableWaterRetention->removeRow(row);
    mySoil->horizon[currentHorizon].dbData.waterRetention.erase(mySoil->horizon[currentHorizon].dbData.waterRetention.begin() + row);
    std::string errorString;
    soil::setHorizon(&(mySoil->horizon[currentHorizon]), myTextureClassList, myFittingOptions, errorString);

    emit updateSignal();
}

void TabWaterRetentionData::resetAll()
{
    // delete all Widgets
    barHorizons.clear();
    deleteRow->setEnabled(false);
    tableWaterRetention->clearContents();
    tableWaterRetention->setRowCount(0);
    tableWaterRetention->clearSelection();
    fillData = false;
}

void TabWaterRetentionData::resetTable()
{
    deleteRow->setEnabled(false);
    tableWaterRetention->clearContents();
    tableWaterRetention->setRowCount(0);
    tableWaterRetention->clearSelection();
}

void TabWaterRetentionData::cellClicked(int row, int column)
{

    tableWaterRetention->clearSelection();
    tableWaterRetention->setSelectionBehavior(QAbstractItemView::SelectItems);
    tableWaterRetention->item(row,column)->setSelected(true);
    deleteRow->setEnabled(false);
}

void TabWaterRetentionData::cellChanged(int row, int column)
{

    tableWaterRetention->item(row,column)->setSelected(true);
    if (tableWaterRetention->itemAt(row,column) == nullptr)
    {
        return;
    }
    tableWaterRetention->blockSignals(true);
    QString data = tableWaterRetention->item(row, column)->text();
    data.replace(",", ".");
    // set new value
    switch (column) {
        case 0:
        {
        // water potential
            if (data.toDouble() < 0)
            {
                QMessageBox::critical(nullptr, "Error!", "Insert valid water potential");
                if (row == 0)
                {
                    tableWaterRetention->item(row, column)->setText(QString::number(0));
                }
                else
                {
                    tableWaterRetention->item(row, column)->setText(tableWaterRetention->item(row-1,column)->text());
                }
            }
            else if (data.toDouble() < 1)
            {
                tableWaterRetention->item(row, column)->setText(QString::number(data.toDouble(), 'f', 3));
            }
            else
            {
                tableWaterRetention->item(row, column)->setText(QString::number(data.toDouble(), 'f', 1));
            }
            mySoil->horizon[currentHorizon].dbData.waterRetention[row].water_potential = data.toDouble();
            break;
        }
        // water content
        case 1:
        {
            if (data.toDouble() < 0 || data.toDouble() > 1)
            {
                QMessageBox::critical(nullptr, "Error!", "Insert valid water content");
                if (row == 0)
                {
                    tableWaterRetention->item(row, column)->setText(QString::number(0));
                }
                else
                {
                    tableWaterRetention->item(row, column)->setText(tableWaterRetention->item(row-1,column)->text());
                }
            }
            else
            {
                tableWaterRetention->item(row, column)->setText(QString::number(data.toFloat(), 'f', 3));
            }
            mySoil->horizon[currentHorizon].dbData.waterRetention[row].water_content = data.toDouble();
            break;
        }

    }

    tableWaterRetention->sortByColumn(0, Qt::AscendingOrder);
    sort(mySoil->horizon[currentHorizon].dbData.waterRetention.begin(), mySoil->horizon[currentHorizon].dbData.waterRetention.end(), soil::sortWaterPotential);

    std::string errorString;
    soil::setHorizon(&(mySoil->horizon[currentHorizon]), myTextureClassList, myFittingOptions, errorString);

    tableWaterRetention->update();
    tableWaterRetention->blockSignals(false);
    if (!horizonChanged.contains(currentHorizon))
    {
        horizonChanged << currentHorizon;
    }
    emit updateSignal();

}


void TabWaterRetentionData::resetHorizonChanged()
{
    horizonChanged.clear();
}

QVector<int> TabWaterRetentionData::getHorizonChanged() const
{
    return horizonChanged;
}

bool TabWaterRetentionData::getFillData() const
{
    return fillData;
}

void TabWaterRetentionData::setFillData(bool value)
{
    fillData = value;
}

void TabWaterRetentionData::widgetClicked(int index)
{
    // check selection state
    if (barHorizons.barList[index]->getSelected())
    {
        // clear previous selection
        barHorizons.deselectAll(index);
        //select the right
        barHorizons.selectItem(index);
        addRow->setEnabled(true);
    }
    else
    {
        resetTable();
        addRow->setEnabled(false);
        currentHorizon = -1;
        emit horizonSelected(-1);
        return;
    }

    resetTable();
    int row = 0;
    currentHorizon = index;

    if (mySoil->horizon[index].dbData.waterRetention.size() != 0)
    {
        row = row + int(mySoil->horizon[index].dbData.waterRetention.size());
    }
    tableWaterRetention->setRowCount(row);
    tableWaterRetention->setSortingEnabled(false);
    tableWaterRetention->blockSignals(true);

    for (unsigned int j = 0; j < mySoil->horizon[index].dbData.waterRetention.size(); j++)
    {
        if (mySoil->horizon[index].dbData.waterRetention[j].water_potential < 1)
        {
            tableWaterRetention->setItem(j, 0, new Crit3DTableWidgetItem( QString::number(mySoil->horizon[index].dbData.waterRetention[j].water_potential, 'f', 3)));
        }
        else
        {
            tableWaterRetention->setItem(j, 0, new Crit3DTableWidgetItem( QString::number(mySoil->horizon[index].dbData.waterRetention[j].water_potential, 'f', 1)));
        }
        tableWaterRetention->item(j,0)->setTextAlignment(Qt::AlignRight);

        tableWaterRetention->setItem(j, 1, new Crit3DTableWidgetItem( QString::number(mySoil->horizon[index].dbData.waterRetention[j].water_content, 'f', 3 )));
        tableWaterRetention->item(j,1)->setTextAlignment(Qt::AlignRight);
    }

    tableWaterRetention->setSortingEnabled(true);
    tableWaterRetention->sortByColumn(0, Qt::AscendingOrder);
    tableWaterRetention->blockSignals(false);

    emit horizonSelected(index);

}
