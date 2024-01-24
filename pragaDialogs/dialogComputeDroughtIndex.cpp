#include "dialogComputeDroughtIndex.h"

DialogComputeDroughtIndex::DialogComputeDroughtIndex(bool isMeteoGridLoaded, bool isMeteoPointLoaded, QDate myDatePointsFrom, QDate myDatePointsTo, QDate myDateGridFrom, QDate myDateGridTo)
    : isMeteoGridLoaded(isMeteoGridLoaded), isMeteoPointLoaded(isMeteoPointLoaded)
{

    QVBoxLayout* mainLayout = new QVBoxLayout;
    QHBoxLayout targetLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
    QVBoxLayout *indexLayout = new QVBoxLayout;

    QLabel *subTitleLabel = new QLabel();
    QGroupBox *targetGroupBox = new QGroupBox("Target");
    pointsButton.setText("meteo points");
    gridButton.setText("meteo grid");

    if (isMeteoPointLoaded)
    {
        pointsButton.setEnabled(true);
        if (!isMeteoGridLoaded)
        {
            pointsButton.setChecked(true);
        }
    }
    else
    {
        pointsButton.setEnabled(false);
    }
    if (isMeteoGridLoaded)
    {
        gridButton.setEnabled(true);
        if (!isMeteoPointLoaded)
        {
            gridButton.setChecked(true);
        }
    }
    else
    {
        gridButton.setEnabled(false);
    }

    if (isMeteoPointLoaded && isMeteoGridLoaded)
    {
        // default grid
        gridButton.setChecked(true);
        pointsButton.setChecked(false);
    }

    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        dateFrom.setDate(myDatePointsFrom);
        dateTo.setDate(myDatePointsTo);
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        dateFrom.setDate(myDateGridFrom);
        dateTo.setDate(myDateGridTo);
    }

    targetLayout.addWidget(&pointsButton);
    targetLayout.addWidget(&gridButton);
    targetGroupBox->setLayout(&targetLayout);

    mainLayout->addWidget(subTitleLabel);
    QLabel *dateFromLabel = new QLabel(tr("Reference Period From"));
    dateLayout->addWidget(dateFromLabel);
    dateLayout->addWidget(&dateFrom);
    QLabel *dateToLabel = new QLabel(tr("Reference Period To"));
    dateLayout->addWidget(dateToLabel);
    dateLayout->addWidget(&dateTo);


    dateFrom.setDisplayFormat("dd.MM.yyyy");
    dateTo.setDisplayFormat("dd.MM.yyyy");

    QLabel *variableLabel = new QLabel(tr("Drought Index: "));
    std::map<meteoVariable, std::string>::const_iterator it;
    listIndex.setSelectionMode(QAbstractItemView::SingleSelection);
    listIndex.addItem("INDEX_SPI");
    listIndex.addItem("INDEX_SPEI");
    listIndex.addItem("INDEX_DECILES");

    indexLayout->addWidget(variableLabel);
    indexLayout->addWidget(&listIndex);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });
    connect(&listIndex, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->indexClicked(item); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addWidget(targetGroupBox);
    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(indexLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogComputeDroughtIndex::~DialogComputeDroughtIndex()
{
    close();
}


void DialogComputeDroughtIndex::indexClicked(QListWidgetItem* item)
{
    Q_UNUSED(item);
}

void DialogComputeDroughtIndex::done(bool res)
{
    if (res) // ok
    {
        if (listIndex.selectedItems().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing drought index", "Select a drought index");
            return;
        }
        if (dateFrom.date() > dateTo.date())
        {
            QMessageBox::information(nullptr, "Invalid interval", "First date should be <= last date ");
            return;
        }
        QDialog::done(QDialog::Accepted);
        return;
    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}

QString DialogComputeDroughtIndex::getIndex() const
{
    return listIndex.currentItem()->text();
}

QDate DialogComputeDroughtIndex::getDateFrom() const
{
    return dateFrom.date();
}

QDate DialogComputeDroughtIndex::getDateTo() const
{
    return dateTo.date();
}

