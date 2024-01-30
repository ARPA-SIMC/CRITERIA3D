#include "dialogComputeDroughtIndex.h"

DialogComputeDroughtIndex::DialogComputeDroughtIndex(bool isMeteoGridLoaded, bool isMeteoPointLoaded, int yearPointsFrom, int yearPointsTo, int yearGridFrom, int yearGridTo, QDate currentDate)
    : isMeteoGridLoaded(isMeteoGridLoaded), isMeteoPointLoaded(isMeteoPointLoaded),
    yearPointsFrom(yearPointsFrom), yearPointsTo(yearPointsTo), yearGridFrom(yearGridFrom), yearGridTo(yearGridTo), currentDate(currentDate)
{

    QVBoxLayout* mainLayout = new QVBoxLayout;
    QHBoxLayout targetLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
    QHBoxLayout *groupLayout = new QHBoxLayout;
    groupLayout->setAlignment(Qt::AlignTop);
    QVBoxLayout *indexLayout = new QVBoxLayout;
    QHBoxLayout *timescaleLayout = new QHBoxLayout;

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
        yearFrom.setText(QString("%1").arg(yearPointsFrom));
        yearTo.setText(QString("%1").arg(yearPointsTo));
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        yearFrom.setText(QString("%1").arg(yearGridFrom));
        yearTo.setText(QString("%1").arg(yearGridTo));
    }
    yearFrom.setValidator(new QIntValidator(100, 3000));
    yearTo.setValidator(new QIntValidator(100, 3000));
    yearFrom.setMaximumWidth(this->width()/9);
    yearTo.setMaximumWidth(this->width()/9);
    targetLayout.addWidget(&pointsButton);
    targetLayout.addWidget(&gridButton);
    targetGroupBox->setLayout(&targetLayout);

    mainLayout->addWidget(subTitleLabel);
    QLabel *yearFromLabel = new QLabel(tr("Reference Start Year:"));
    dateLayout->addWidget(yearFromLabel);
    dateLayout->addWidget(&yearFrom);
    QLabel *yearToLabel = new QLabel(tr("Reference End Year:"));
    dateLayout->addWidget(yearToLabel);
    dateLayout->addWidget(&yearTo);
    myDate.setDate(currentDate);
    QLabel *dateToLabel = new QLabel(tr("Date:"));
    dateLayout->addWidget(dateToLabel);
    dateLayout->addWidget(&myDate);

    QLabel *variableLabel = new QLabel(tr("Drought Index: "));
    std::map<meteoVariable, std::string>::const_iterator it;
    listIndex.setSelectionMode(QAbstractItemView::SingleSelection);
    listIndex.addItem("INDEX_SPI");
    listIndex.addItem("INDEX_SPEI");
    listIndex.addItem("INDEX_DECILES");

    timescaleLabel.setText("Timescale: ");
    timescaleList.addItem("3");
    timescaleList.addItem("6");
    timescaleList.addItem("12");
    timescaleList.addItem("24");

    variableLabel->setMaximumWidth(this->width()/2);
    listIndex.setMaximumWidth(this->width()/2);
    listIndex.setMaximumHeight(this->height()/3);
    indexLayout->addWidget(variableLabel);
    indexLayout->addWidget(&listIndex);

    timescaleLabel.setMaximumWidth(this->width()/2);
    timescaleLayout->addWidget(&timescaleLabel);
    timescaleLayout->addWidget(&timescaleList);
    timescaleLabel.setVisible(false);
    timescaleList.setVisible(false);

    groupLayout->addLayout(indexLayout);
    groupLayout->addLayout(timescaleLayout);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });
    connect(&pointsButton, &QRadioButton::clicked, [=](){ this->targetChange(); });
    connect(&gridButton, &QRadioButton::clicked, [=](){ this->targetChange(); });
    connect(&listIndex, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->indexClicked(item); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addWidget(targetGroupBox);
    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(groupLayout);
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
    if (item->text() == "INDEX_SPI" || item->text() == "INDEX_SPEI")
    {
        timescaleLabel.setVisible(true);
        timescaleList.setVisible(true);
    }
    else if (item->text() == "INDEX_DECILES")
    {
        timescaleLabel.setVisible(false);
        timescaleList.setVisible(false);
    }
}

void DialogComputeDroughtIndex::targetChange()
{
    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        yearFrom.setText(QString("%1").arg(yearPointsFrom));
        yearTo.setText(QString("%1").arg(yearPointsTo));

    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        yearFrom.setText(QString("%1").arg(yearGridFrom));
        yearTo.setText(QString("%1").arg(yearGridTo));

    }
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
        if (yearFrom.text().toInt() > yearTo.text().toInt())
        {
            QMessageBox::information(nullptr, "Invalid interval", "First reference year should be <= last reference year ");
            return;
        }
        if (!isMeteoGrid)
        {
            if (yearFrom.text().toInt() < yearPointsFrom || yearTo.text().toInt() > yearPointsTo)
            {
                QMessageBox::information(nullptr, "Invalid interval", QString("Meteo Point reference years outside interval %1 - %2").arg(yearPointsFrom).arg(yearPointsTo));
                return;
            }
        }
        else
        {
            if (yearFrom.text().toInt() < yearGridFrom || yearTo.text().toInt() > yearGridTo)
            {
                QMessageBox::information(nullptr, "Invalid interval", QString("Meteo Grid reference years outside interval %1 - %2").arg(yearGridFrom).arg(yearGridTo));
                return;
            }
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

int DialogComputeDroughtIndex::getYearFrom() const
{
    return yearFrom.text().toInt();
}

int DialogComputeDroughtIndex::getYearTo() const
{
    return yearTo.text().toInt();
}

int DialogComputeDroughtIndex::getTimescale() const
{
    return timescaleList.currentText().toInt();
}

QDate DialogComputeDroughtIndex::getDate() const
{
    return myDate.date();
}

bool DialogComputeDroughtIndex::getIsMeteoGrid() const
{
    return isMeteoGrid;
}

