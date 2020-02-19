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


#include "cropWidget.h"
#include "dialogNewCrop.h"
#include "cropDbTools.h"
#include "dbMeteoCriteria1D.h"
#include "utilities.h"

#include <QFileInfo>
#include <QFileDialog>
#include <QMessageBox>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QDate>

#include <QDebug>


Crit3DCropWidget::Crit3DCropWidget()
{
    this->setWindowTitle(QStringLiteral("Crop"));
    this->resize(1240, 700);

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *saveButtonLayout = new QHBoxLayout();
    QHBoxLayout *cropLayout = new QHBoxLayout();
    QVBoxLayout *infoLayout = new QVBoxLayout();
    QGridLayout *cropInfoLayout = new QGridLayout();
    QGridLayout *meteoInfoLayout = new QGridLayout();
    QGridLayout *parametersLaiLayout = new QGridLayout();
    QGridLayout *parametersRootDepthLayout = new QGridLayout();

    // check save button pic
    QString docPath, saveButtonPath, updateButtonPath;
    if (searchDocPath(&docPath))
    {
        saveButtonPath = docPath + "img/saveButton.png";
        updateButtonPath = docPath + "img/updateButton.png";
    }
    else
    {
        saveButtonPath = "../img/saveButton.png";
        updateButtonPath = "../img/updateButton.png";
    }

    QPixmap savePixmap(saveButtonPath);
    QPixmap updatePixmap(updateButtonPath);
    QPushButton *saveButton = new QPushButton();
    QPushButton *updateButton = new QPushButton();
    QIcon saveButtonIcon(savePixmap);
    QIcon updateButtonIcon(updatePixmap);
    saveButton->setIcon(saveButtonIcon);
    saveButton->setIconSize(savePixmap.rect().size());
    saveButton->setFixedSize(savePixmap.rect().size());

    saveButtonLayout->setAlignment(Qt::AlignLeft);
    saveButtonLayout->addWidget(saveButton);

    updateButton->setIcon(updateButtonIcon);
    updateButton->setIconSize(savePixmap.rect().size());
    updateButton->setFixedSize(savePixmap.rect().size());

    saveButtonLayout->setAlignment(Qt::AlignLeft);
    saveButtonLayout->addWidget(updateButton);

    QLabel *cropName = new QLabel(tr("CROP_NAME: "));

    QLabel *cropId = new QLabel(tr("ID_CROP: "));
    cropIdValue = new QLineEdit();
    cropIdValue->setReadOnly(true);

    QLabel * cropType= new QLabel(tr("crop type: "));
    cropTypeValue = new QLineEdit();
    cropTypeValue->setReadOnly(true);

    cropSowingValue = new QSpinBox();
    cropCycleMaxValue = new QSpinBox();
    cropSowingValue->setMinimum(-365);
    cropSowingValue->setMaximum(365);
    cropCycleMaxValue->setMinimum(0);
    cropCycleMaxValue->setMaximum(365);
    cropSowing.setText("sowing DOY: ");
    cropCycleMax.setText("cycle max duration: ");

    QLabel * maxKc= new QLabel(tr("max kc [-]: "));
    maxKcValue = new QLineEdit();

    infoCropGroup = new QGroupBox(tr(""));
    infoMeteoGroup = new QGroupBox(tr(""));
    laiParametersGroup = new QGroupBox(tr(""));
    rootParametersGroup = new QGroupBox(tr(""));

    infoCropGroup->setFixedWidth(this->width()/4.5);
    infoMeteoGroup->setFixedWidth(this->width()/4.5);
    laiParametersGroup->setFixedWidth(this->width()/4.5);
    rootParametersGroup->setFixedWidth(this->width()/4.5);

    infoCropGroup->setTitle("Crop");
    infoMeteoGroup->setTitle("Meteo");
    laiParametersGroup->setTitle("LAI Parameters");
    rootParametersGroup->setTitle("root depth Parameters");

    cropInfoLayout->addWidget(cropName, 0, 0);
    cropInfoLayout->addWidget(&cropListComboBox, 0, 1);
    cropInfoLayout->addWidget(cropId, 1, 0);
    cropInfoLayout->addWidget(cropIdValue, 1, 1);
    cropInfoLayout->addWidget(cropType, 2, 0);
    cropInfoLayout->addWidget(cropTypeValue, 2, 1);
    cropInfoLayout->addWidget(&cropSowing, 3, 0);
    cropInfoLayout->addWidget(cropSowingValue, 3, 1);
    cropInfoLayout->addWidget(&cropCycleMax, 4, 0);
    cropInfoLayout->addWidget(cropCycleMaxValue, 4, 1);
    cropInfoLayout->addWidget(maxKc, 5, 0);
    cropInfoLayout->addWidget(maxKcValue, 5, 1);

    QLabel *meteoName = new QLabel(tr("METEO_NAME: "));

    QLabel *meteoYear = new QLabel(tr("year: "));

    QLabel *lat = new QLabel(tr("latitude: "));
    latValue = new QDoubleSpinBox();
    latValue->setMinimum(-90);
    latValue->setMaximum(90);
    latValue->setDecimals(3);

    QLabel *lon = new QLabel(tr("longitude: "));
    lonValue = new QDoubleSpinBox();
    lonValue->setMinimum(-180);
    lonValue->setMaximum(180);
    lonValue->setDecimals(3);

    meteoInfoLayout->addWidget(meteoName, 0, 0);
    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 1);
    meteoInfoLayout->addWidget(meteoYear, 1, 0);
    meteoInfoLayout->addWidget(&yearListComboBox, 1, 1);
    meteoInfoLayout->addWidget(lat, 2, 0);
    meteoInfoLayout->addWidget(latValue, 2, 1);
    meteoInfoLayout->addWidget(lon, 3, 0);
    meteoInfoLayout->addWidget(lonValue, 3, 1);

    QLabel *LAImin = new QLabel(tr("LAI min [m2 m-2]: "));
    LAIminValue = new QLineEdit();

    QLabel *LAImax = new QLabel(tr("LAI max [m2 m-2]: "));
    LAImaxValue = new QLineEdit();

    LAIgrass = new QLabel(tr("LAI grass [m2 m-2]: "));
    LAIgrassValue = new QLineEdit();

    QLabel *thermalThreshold = new QLabel(tr("thermal threshold [°C]: "));
    thermalThresholdValue = new QLineEdit();

    QLabel *upperThermalThreshold = new QLabel(tr("upper thermal threshold [°C]: "));
    upperThermalThresholdValue = new QLineEdit();

    QLabel *degreeDaysEmergence = new QLabel(tr("degree days emergence [°C]: "));
    degreeDaysEmergenceValue = new QLineEdit();

    QLabel *degreeDaysLAIinc = new QLabel(tr("degree days LAI increase [°C]: "));
    degreeDaysLAIincValue = new QLineEdit();

    QLabel *degreeDaysLAIdec = new QLabel(tr("degree days LAI decrease [°C]: "));
    degreeDaysLAIdecValue = new QLineEdit();

    QLabel *LAIcurveA = new QLabel(tr("LAI curve factor A [-]: "));
    LAIcurveAValue = new QLineEdit();

    QLabel *LAIcurveB = new QLabel(tr("LAI curve factor B [-]: "));
    LAIcurveBValue = new QLineEdit();

    parametersLaiLayout->addWidget(LAImin, 0, 0);
    parametersLaiLayout->addWidget(LAIminValue, 0, 1);
    parametersLaiLayout->addWidget(LAImax, 1, 0);
    parametersLaiLayout->addWidget(LAImaxValue, 1, 1);
    parametersLaiLayout->addWidget(LAIgrass, 3, 0);
    parametersLaiLayout->addWidget(LAIgrassValue, 3, 1);
    parametersLaiLayout->addWidget(thermalThreshold, 4, 0);
    parametersLaiLayout->addWidget(thermalThresholdValue, 4, 1);
    parametersLaiLayout->addWidget(upperThermalThreshold, 5, 0);
    parametersLaiLayout->addWidget(upperThermalThresholdValue, 5, 1);
    parametersLaiLayout->addWidget(degreeDaysEmergence, 6, 0);
    parametersLaiLayout->addWidget(degreeDaysEmergenceValue, 6, 1);
    parametersLaiLayout->addWidget(degreeDaysLAIinc, 7, 0);
    parametersLaiLayout->addWidget(degreeDaysLAIincValue, 7, 1);
    parametersLaiLayout->addWidget(degreeDaysLAIdec, 8, 0);
    parametersLaiLayout->addWidget(degreeDaysLAIdecValue, 8, 1);
    parametersLaiLayout->addWidget(LAIcurveA, 9, 0);
    parametersLaiLayout->addWidget(LAIcurveAValue, 9, 1);
    parametersLaiLayout->addWidget(LAIcurveB, 10, 0);
    parametersLaiLayout->addWidget(LAIcurveBValue, 10, 1);

    QLabel *rootDepthZero = new QLabel(tr("root depth zero [m]: "));
    rootDepthZeroValue = new QLineEdit();

    QLabel *rootDepthMax = new QLabel(tr("root depth max [m]: "));
    rootDepthMaxValue = new QLineEdit();

    QLabel *rootShape = new QLabel(tr("root shape: "));
    rootShapeComboBox = new QComboBox();
    rootShapeComboBox->setMaximumWidth(rootParametersGroup->width()/3);

    for (int i=0; i<numRootDistributionType; i++)
    {
        rootDistributionType type = (rootDistributionType) i;
        rootShapeComboBox->addItem(QString::fromStdString(root::getRootDistributionTypeString(type)));
    }

    QLabel *shapeDeformation = new QLabel(tr("shape deformation [-]: "));
    shapeDeformationValue = new QLineEdit();

    degreeDaysInc = new QLabel(tr("degree days root inc [°C]: "));
    degreeDaysIncValue = new QLineEdit();

    parametersRootDepthLayout->addWidget(rootDepthZero, 0, 0);
    parametersRootDepthLayout->addWidget(rootDepthZeroValue, 0, 1);
    parametersRootDepthLayout->addWidget(rootDepthMax, 1, 0);
    parametersRootDepthLayout->addWidget(rootDepthMaxValue, 1, 1);
    parametersRootDepthLayout->addWidget(rootShape, 2, 0);
    parametersRootDepthLayout->addWidget(rootShapeComboBox, 2, 1);
    parametersRootDepthLayout->addWidget(shapeDeformation, 3, 0);
    parametersRootDepthLayout->addWidget(shapeDeformationValue, 3, 1);
    parametersRootDepthLayout->addWidget(degreeDaysInc, 4, 0);
    parametersRootDepthLayout->addWidget(degreeDaysIncValue, 4, 1);


    infoCropGroup->setLayout(cropInfoLayout);
    infoMeteoGroup->setLayout(meteoInfoLayout);
    laiParametersGroup->setLayout(parametersLaiLayout);
    rootParametersGroup->setLayout(parametersRootDepthLayout);

    infoLayout->addWidget(infoCropGroup);
    infoLayout->addWidget(infoMeteoGroup);
    infoLayout->addWidget(laiParametersGroup);
    infoLayout->addWidget(rootParametersGroup);

    mainLayout->addLayout(saveButtonLayout);
    mainLayout->addLayout(cropLayout);
    mainLayout->setAlignment(Qt::AlignTop);

    cropLayout->addLayout(infoLayout);
    tabWidget = new QTabWidget;
    tabLAI = new TabLAI();
    tabRootDepth = new TabRootDepth();
    tabWidget->addTab(tabLAI, tr("LAI development"));
    tabWidget->addTab(tabRootDepth, tr("Root depth"));
    cropLayout->addWidget(tabWidget);

    this->setLayout(mainLayout);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *fileMenu = new QMenu("File");
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(fileMenu);
    menuBar->addMenu(editMenu);
    this->layout()->setMenuBar(menuBar);

    QAction* openCropDB = new QAction(tr("&Open dbCrop"), this);
    QAction* openMeteoDB = new QAction(tr("&Open dbMeteo"), this);
    saveChanges = new QAction(tr("&Save Changes"), this);

    QAction* newCrop = new QAction(tr("&New Crop"), this);
    QAction* deleteCrop = new QAction(tr("&Delete Crop"), this);
    restoreData = new QAction(tr("&Restore Data"), this);

    fileMenu->addAction(openCropDB);
    fileMenu->addAction(openMeteoDB);
    fileMenu->addAction(saveChanges);

    editMenu->addAction(newCrop);
    editMenu->addAction(deleteCrop);
    editMenu->addAction(restoreData);

    myCrop = nullptr;
    meteoPoint = nullptr;
    nrLayers = 100;     // depth each layer = 2 cm
    totalSoilDepth = 2; // [m] average soil depth

    connect(openCropDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenCropDB);
    connect(&cropListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseCrop);

    connect(openMeteoDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenMeteoDB);
    connect(&meteoListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseMeteo);
    connect(&yearListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseYear);

    connect(tabWidget, &QTabWidget::currentChanged, [=](int index){ this->tabChanged(index); });

    connect(newCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionNewCrop);
    connect(deleteCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionDeleteCrop);
    connect(restoreData, &QAction::triggered, this, &Crit3DCropWidget::on_actionRestoreData);

    //set current tab
    tabChanged(0);
}


void Crit3DCropWidget::on_actionOpenCropDB()
{
    QString dbCropName = QFileDialog::getOpenFileName(this, tr("Open crop database"), "", tr("SQLite files (*.db)"));
    if (dbCropName == "")
    {
        return;
    }

    // open crop db
    QString error;
    if (! openDbCrop(dbCropName, &dbCrop, &error))
    {
        QMessageBox::critical(nullptr, "Error DB crop", error);
        return;
    }

    // read crop list
    QStringList cropStringList;
    if (! getCropNameList(&dbCrop, &cropStringList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // show crop list
    this->cropListComboBox.clear();
    for (int i = 0; i < cropStringList.size(); i++)
    {
        this->cropListComboBox.addItem(cropStringList[i]);
    }

    //saveChanges->setEnabled(true);
}


void Crit3DCropWidget::on_actionOpenMeteoDB()
{
    QString dbMeteoName = QFileDialog::getOpenFileName(this, tr("Open meteo database"), "", tr("SQLite files (*.db)"));
    if (dbMeteoName == "")
    {
        return;
    }

    // open meteo db
    QString error;
    if (! openDbMeteo(dbMeteoName, &dbMeteo, &error))
    {
        QMessageBox::critical(nullptr, "Error DB meteo", error);
        return;
    }

    // read id_meteo list
    QStringList idMeteoList;
    if (! getMeteoPointList(&dbMeteo, &idMeteoList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // show id_meteo list
    this->meteoListComboBox.clear();
    for (int i = 0; i < idMeteoList.size(); i++)
    {
        this->meteoListComboBox.addItem(idMeteoList[i]);
    }
    //saveChanges->setEnabled(true);
}


void Crit3DCropWidget::on_actionChooseCrop(QString cropName)
{
    QString error;
    QString idCrop = getIdCropFromName(&dbCrop, cropName, &error);
    if (idCrop.isEmpty())
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // delete previous crop
    if (myCrop != nullptr)
    {
        delete myCrop;
    }
    myCrop = new Crit3DCrop();

    cropIdValue->setText(idCrop);
    if (!loadCropParameters(idCrop, myCrop, &dbCrop, &error))
    {
        if (error.contains("Empty"))
        {
            QMessageBox::information(nullptr, "Warning", error);
        }
        else
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }

    }
    cropTypeValue->setText(QString::fromStdString(getCropTypeString(myCrop->type)));

    if (! myCrop->isPluriannual())
    {
        cropSowing.setVisible(true);
        cropCycleMax.setVisible(true);
        cropSowingValue->setValue(myCrop->sowingDoy);
        cropSowingValue->setVisible(true);
        cropCycleMaxValue->setValue(myCrop->plantCycle);
        cropCycleMaxValue->setVisible(true);
    }
    else
    {
        cropSowing.setVisible(false);
        cropCycleMax.setVisible(false);
        cropSowingValue->setVisible(false);
        cropCycleMaxValue->setVisible(false);
    }   
    maxKcValue->setText(QString::number(myCrop->kcMax));

    // LAI parameters
    LAIminValue->setText(QString::number(myCrop->LAImin));
    LAImaxValue->setText(QString::number(myCrop->LAImax));
    if (myCrop->type == FRUIT_TREE)
    {
        LAIgrass->setVisible(true);
        LAIgrassValue->setVisible(true);
        LAIgrassValue->setText(QString::number(myCrop->LAIgrass));
    }
    else
    {
        LAIgrass->setVisible(false);
        LAIgrassValue->setVisible(false);
    }
    thermalThresholdValue->setText(QString::number(myCrop->thermalThreshold));
    upperThermalThresholdValue->setText(QString::number(myCrop->upperThermalThreshold));
    degreeDaysEmergenceValue->setText(QString::number(myCrop->degreeDaysEmergence));
    degreeDaysLAIincValue->setText(QString::number(myCrop->degreeDaysIncrease));
    degreeDaysLAIdecValue->setText(QString::number(myCrop->degreeDaysDecrease));
    LAIcurveAValue->setText(QString::number(myCrop->LAIcurve_a));
    LAIcurveBValue->setText(QString::number(myCrop->LAIcurve_b));

    // root parameters
    rootDepthZeroValue->setText(QString::number(myCrop->roots.rootDepthMin));
    rootDepthMaxValue->setText(QString::number(myCrop->roots.rootDepthMax));
    shapeDeformationValue->setText(QString::number(myCrop->roots.shapeDeformation));
    if (myCrop->isPluriannual())
    {
        degreeDaysInc->setVisible(false);
        degreeDaysIncValue->setVisible(false);
    }
    else
    {
        degreeDaysInc->setVisible(true);
        degreeDaysIncValue->setVisible(true);
        degreeDaysIncValue->setText(QString::number(myCrop->roots.degreeDaysRootGrowth));
    }

    if (meteoPoint != nullptr && !yearListComboBox.currentText().isEmpty())
    {
        updateTabLAI();
    }
}


void Crit3DCropWidget::on_actionChooseMeteo(QString idMeteo)
{
    QString error, lat, lon;

    if (getLatLonFromIdMeteo(&dbMeteo, idMeteo, &lat, &lon, &error))
    {
        latValue->setValue(lat.toDouble());
        lonValue->setValue(lon.toDouble());
    }

    tableMeteo = getTableNameFromIdMeteo(&dbMeteo, idMeteo, &error);

    QStringList yearList;
    if (!getYearList(&dbMeteo, tableMeteo, &yearList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        this->yearListComboBox.clear();
        return;
    }

    int pos = 0;
    for (int i = 0; i<yearList.size(); i++)
    {
        if ( !checkYear(&dbMeteo, tableMeteo, yearList[i], &error))
        {
            yearList.removeAt(pos);
        }
        else
        {
            pos = pos + 1;
        }
    }

    // add year if exists previous year
    for (int i = 1; i<yearList.size(); i++)
    {
        if (yearList[i].toInt() == yearList[i-1].toInt()+1)
        {
            this->yearListComboBox.addItem(yearList[i]);
        }
    }

}


void Crit3DCropWidget::on_actionChooseYear(QString year)
{
    QString error;

    // delete previous meteoPoint
    if (meteoPoint != nullptr)
    {
        delete meteoPoint;
    }
    meteoPoint = new Crit3DMeteoPoint();
    meteoPoint->latitude = latValue->text().toDouble();
    meteoPoint->longitude = lonValue->text().toDouble();

    QDate firstDate(year.toInt(), 1, 1);
    int daysInYear = firstDate.daysInYear();
    meteoPoint->initializeObsDataD(daysInYear, getCrit3DDate(firstDate));

    if (!fillDailyTempCriteria1D(&dbMeteo, tableMeteo, meteoPoint, year, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }
    if (myCrop != nullptr)
    {
        updateTabLAI();
    }

}


void Crit3DCropWidget::on_actionDeleteCrop()
{
    QString msg;
        if (cropListComboBox.currentText().isEmpty())
        {
            msg = "Select the soil to be deleted";
            QMessageBox::information(nullptr, "Warning", msg);
        }
        else
        {
            QMessageBox::StandardButton confirm;
            msg = "Are you sure you want to delete "+cropListComboBox.currentText()+" ?";
            confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::No);
            QString error;

            if (confirm == QMessageBox::Yes)
            {
                if (deleteCropData(&dbCrop, cropListComboBox.currentText(), &error))
                {
                    cropListComboBox.removeItem(cropListComboBox.currentIndex());
                }
            }
            else
            {
                return;
            }
        }
}


void Crit3DCropWidget::on_actionRestoreData()
{
    // TO DO
    // deve salvare il db Crop originario? le modifiche possibili sono solo dovute a delete/new crop?
}


void Crit3DCropWidget::on_actionNewCrop()
{
    if (!dbCrop.isOpen())
    {
        QString msg = "Open a Db Crop";
        QMessageBox::information(nullptr, "Warning", msg);
        return;
    }
    Crit3DCrop* newCrop = new Crit3DCrop();
    DialogNewCrop dialog(newCrop);
    QString error;
    if (dialog.result() != QDialog::Accepted)
    {
        delete newCrop;
        return;
    }
    else
    {
        // TO DO
        // write newCrop on Db
        delete newCrop;
    }
}

void Crit3DCropWidget::updateTabLAI()
{
    int currentDoy = 1;
    if (myCrop == nullptr || meteoPoint == nullptr)
    {
        return;
    }
    tabLAI->computeLAI(myCrop, meteoPoint, yearListComboBox.currentText().toInt(), nrLayers, totalSoilDepth, currentDoy);
}

void Crit3DCropWidget::tabChanged(int index)
{

    if (index == 0) //LAI tab
    {
        rootParametersGroup->hide();
        laiParametersGroup->setVisible(true);

    }
    else if(index == 1) //root depth tab
    {
        laiParametersGroup->hide();
        rootParametersGroup->setVisible(true);
    }
}
