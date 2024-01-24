#include "dialogComputeDroughtIndex.h"

DialogComputeDroughtIndex::DialogComputeDroughtIndex(bool isMeteoGridLoaded, bool isMeteoPointLoaded, int yearPointsFrom, int yearPointsTo, int yearGridFrom, int yearGridTo)
    : isMeteoGridLoaded(isMeteoGridLoaded), isMeteoPointLoaded(isMeteoPointLoaded),
    yearPointsFrom(yearPointsFrom), yearPointsTo(yearPointsTo), yearGridFrom(yearGridFrom), yearGridTo(yearGridTo)
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
    targetLayout.addWidget(&pointsButton);
    targetLayout.addWidget(&gridButton);
    targetGroupBox->setLayout(&targetLayout);

    mainLayout->addWidget(subTitleLabel);
    QLabel *yearFromLabel = new QLabel(tr("Reference Start Year"));
    dateLayout->addWidget(yearFromLabel);
    dateLayout->addWidget(&yearFrom);
    QLabel *yearToLabel = new QLabel(tr("Reference End Year"));
    dateLayout->addWidget(yearToLabel);
    dateLayout->addWidget(&yearTo);

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
        if (yearFrom.text().toInt() > yearTo.text().toInt())
        {
            QMessageBox::information(nullptr, "Invalid interval", "First reference year should be <= last reference year ");
            return;
        }
        if (pointsButton.isChecked())
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

