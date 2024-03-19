#include "dialogSelectionMeteoPoint.h"

DialogSelectionMeteoPoint::DialogSelectionMeteoPoint(bool active, Crit3DMeteoPointsDbHandler *meteoPointsDbHandler)
    :active(active)
{
    municipalityList = meteoPointsDbHandler->getMunicipalityList();
    provinceList = meteoPointsDbHandler->getProvinceList();
    regionList = meteoPointsDbHandler->getRegionList();
    stateList = meteoPointsDbHandler->getStateList();
    datasetList = meteoPointsDbHandler->getDatasetList();

    setWindowTitle("Select");
    setMinimumWidth(400);
    QVBoxLayout mainLayout;
    QHBoxLayout selectionLayout;
    QHBoxLayout buttonLayout;

    selectionMode.addItem("municipality");
    selectionMode.addItem("province");
    selectionMode.addItem("region");
    selectionMode.addItem("state");
    selectionMode.addItem("dataset");
    selectionMode.addItem("name");
    selectionMode.addItem("id_point");
    selectionMode.addItem("altitude");
    selectionMode.addItem("DEM distance [m]");
    selectionLayout.addWidget(&selectionMode);

    selectionOperation.addItem("=");
    selectionOperation.addItem("!=");
    selectionLayout.addWidget(&selectionOperation);

    selectionItems.addItems(municipalityList);
    selectionItems.setVisible(true);
    itemFromList = true;
    selectionLayout.addWidget(&selectionItems);
    selectionLayout.addWidget(&editItems);
    editItems.setVisible(false);
    editItems.setMaximumWidth(100);
    editItems.setMaximumHeight(30);

    QDialogButtonBox buttonBox;
    QPushButton activeButton;
    if (active)
    {
        activeButton.setText("Active");
    }
    else
    {
        activeButton.setText("Deactive");
    }
    activeButton.setCheckable(true);
    activeButton.setAutoDefault(false);

    QPushButton cancelButton("Cancel");
    cancelButton.setCheckable(true);
    cancelButton.setAutoDefault(false);

    buttonBox.addButton(&activeButton, QDialogButtonBox::AcceptRole);
    buttonBox.addButton(&cancelButton, QDialogButtonBox::RejectRole);

    connect(&selectionMode, &QComboBox::currentTextChanged, [=](){ this->selectionModeChanged(); });
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    buttonLayout.addWidget(&buttonBox);
    mainLayout.addLayout(&selectionLayout);
    mainLayout.addLayout(&buttonLayout);
    setLayout(&mainLayout);

    show();
    exec();
}

void DialogSelectionMeteoPoint::selectionModeChanged()
{
    selectionItems.clear();
    selectionOperation.clear();
    if (selectionMode.currentText() == "municipality")
    {
        selectionItems.addItems(municipalityList);
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        editItems.setVisible(false);
        selectionItems.setVisible(true);
        itemFromList = true;
    }
    else if (selectionMode.currentText() == "province")
    {
        selectionItems.addItems(provinceList);
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        editItems.setVisible(false);
        selectionItems.setVisible(true);
        itemFromList = true;
    }
    else if (selectionMode.currentText() == "region")
    {
        selectionItems.addItems(regionList);
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        editItems.setVisible(false);
        selectionItems.setVisible(true);
        itemFromList = true;
    }
    else if (selectionMode.currentText() == "state")
    {
        selectionItems.addItems(stateList);
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        editItems.setVisible(false);
        selectionItems.setVisible(true);
        itemFromList = true;
    }
    else if (selectionMode.currentText() == "dataset")
    {
        selectionItems.addItems(datasetList);
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        editItems.setVisible(false);
        selectionItems.setVisible(true);
        itemFromList = true;
    }
    else if (selectionMode.currentText() == "name" || selectionMode.currentText() == "id_point")
    {
        selectionOperation.addItem("Like");
        selectionItems.setVisible(false);
        editItems.setVisible(true);
        itemFromList = false;
    }
    else if (selectionMode.currentText() == "altitude" || selectionMode.currentText() == "DEM distance [m]")
    {
        selectionOperation.addItem("=");
        selectionOperation.addItem("!=");
        selectionOperation.addItem(">");
        selectionOperation.addItem("<");
        selectionItems.setVisible(false);
        editItems.setVisible(true);
        itemFromList = false;
    }
}

QString DialogSelectionMeteoPoint::getSelection()
{
    return selectionMode.currentText();
}

QString DialogSelectionMeteoPoint::getOperation()
{
    return selectionOperation.currentText();
}

QString DialogSelectionMeteoPoint::getItem()
{
    if (itemFromList)
    {
        return selectionItems.currentText();
    }
    else
    {
        return editItems.toPlainText();
    }
}

void DialogSelectionMeteoPoint::done(bool res)
{

    if(res)  // ok was pressed
    {
        if (selectionMode.currentText() == "altitude" || selectionMode.currentText() == "DEM distance [m]")
        {
            // value should be a number
            bool ok;
            editItems.toPlainText().toDouble(&ok);
            if (!ok)
            {
                QMessageBox::information(nullptr, "Invalid value", "Enter a numeric value");
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

