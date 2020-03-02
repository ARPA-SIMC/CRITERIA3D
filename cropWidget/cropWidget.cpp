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
#include "criteria1DdbMeteo.h"
#include "soilDbTools.h"
#include "utilities.h"
#include "commonConstants.h"

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
    QHBoxLayout *soilInfoLayout = new QHBoxLayout();
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
    saveButton = new QPushButton();
    updateButton = new QPushButton();
    QIcon saveButtonIcon(savePixmap);
    QIcon updateButtonIcon(updatePixmap);
    saveButton->setIcon(saveButtonIcon);
    saveButton->setIconSize(savePixmap.rect().size());
    saveButton->setFixedSize(savePixmap.rect().size());
    saveButton->setEnabled(false);
    updateButton->setEnabled(false);

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

    infoCropGroup = new QGroupBox(tr(""));
    infoMeteoGroup = new QGroupBox(tr(""));
    infoSoilGroup = new QGroupBox(tr(""));
    laiParametersGroup = new QGroupBox(tr(""));
    rootParametersGroup = new QGroupBox(tr(""));

    infoCropGroup->setFixedWidth(this->width()/4.5);
    infoMeteoGroup->setFixedWidth(this->width()/4.5);
    laiParametersGroup->setFixedWidth(this->width()/4.5);
    rootParametersGroup->setFixedWidth(this->width()/4.5);

    infoCropGroup->setTitle("Crop");
    infoMeteoGroup->setTitle("Meteo");
    infoSoilGroup->setTitle("Soil");
    laiParametersGroup->setTitle("LAI parameters");
    rootParametersGroup->setTitle("root parameters");

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

    QLabel *meteoName = new QLabel(tr("METEO_NAME: "));

    QLabel *meteoYear = new QLabel(tr("year: "));

    QLabel *lat = new QLabel(tr("latitude: "));
    latValue = new QDoubleSpinBox();
    latValue->setMinimum(-90);
    latValue->setMaximum(90);
    latValue->setDecimals(3);

    meteoInfoLayout->addWidget(meteoName, 0, 0);
    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 1);
    meteoInfoLayout->addWidget(meteoYear, 1, 0);
    meteoInfoLayout->addWidget(&yearListComboBox, 1, 1);
    meteoInfoLayout->addWidget(lat, 2, 0);
    meteoInfoLayout->addWidget(latValue, 2, 1);

    soilInfoLayout->addWidget(&soilListComboBox);

    QLabel *LAImin = new QLabel(tr("LAI min [m2 m-2]: "));
    LAIminValue = new QDoubleSpinBox();
    LAIminValue->setMaximumWidth(laiParametersGroup->width()/5);
    LAIminValue->setMinimum(0);
    LAIminValue->setMaximum(10);
    LAIminValue->setDecimals(1);
    LAIminValue->setSingleStep(0.5);

    QLabel *LAImax = new QLabel(tr("LAI max [m2 m-2]: "));
    LAImaxValue = new QDoubleSpinBox();
    LAImaxValue->setMaximumWidth(laiParametersGroup->width()/5);
    LAImaxValue->setMinimum(0);
    LAImaxValue->setMaximum(10);
    LAImaxValue->setDecimals(1);
    LAImaxValue->setSingleStep(0.5);

    LAIgrass = new QLabel(tr("LAI grass [m2 m-2]: "));
    LAIgrassValue = new QLineEdit();
    LAIgrassValue->setMaximumWidth(laiParametersGroup->width()/5);
    QDoubleValidator* firstValidator = new QDoubleValidator(-99.0, 99.0, 3, this);
    QDoubleValidator* secondValidator = new QDoubleValidator(-9999.0, 9999.0, 3, this);
    firstValidator->setNotation(QDoubleValidator::StandardNotation);
    secondValidator->setNotation(QDoubleValidator::StandardNotation);
    LAIgrassValue->setValidator(firstValidator);

    QLabel *thermalThreshold = new QLabel(tr("thermal threshold [°C]: "));
    thermalThresholdValue = new QLineEdit();
    thermalThresholdValue->setMaximumWidth(laiParametersGroup->width()/5);
    thermalThresholdValue->setValidator(firstValidator);

    QLabel *upperThermalThreshold = new QLabel(tr("upper thermal threshold [°C]: "));
    upperThermalThresholdValue = new QLineEdit();
    upperThermalThresholdValue->setMaximumWidth(laiParametersGroup->width()/5);
    upperThermalThresholdValue->setValidator(firstValidator);

    QLabel *degreeDaysEmergence = new QLabel(tr("degree days emergence [°C]: "));
    degreeDaysEmergenceValue = new QLineEdit();
    degreeDaysEmergenceValue->setMaximumWidth(laiParametersGroup->width()/5);
    degreeDaysEmergenceValue->setValidator(secondValidator);

    QLabel *degreeDaysLAIinc = new QLabel(tr("degree days phase 1 [°C]: "));
    degreeDaysLAIincValue = new QLineEdit();
    degreeDaysLAIincValue->setMaximumWidth(laiParametersGroup->width()/5);
    degreeDaysLAIincValue->setValidator(secondValidator);

    QLabel *degreeDaysLAIdec = new QLabel(tr("degree days phase 2 [°C]: "));
    degreeDaysLAIdecValue = new QLineEdit();
    degreeDaysLAIdecValue->setMaximumWidth(laiParametersGroup->width()/5);
    degreeDaysLAIdecValue->setValidator(secondValidator);

    QLabel *LAIcurveA = new QLabel(tr("LAI curve factor A [-]: "));
    LAIcurveAValue = new QLineEdit();
    LAIcurveAValue->setMaximumWidth(laiParametersGroup->width()/5);
    LAIcurveAValue->setValidator(firstValidator);

    QLabel *LAIcurveB = new QLabel(tr("LAI curve factor B [-]: "));
    LAIcurveBValue = new QLineEdit();
    LAIcurveBValue->setMaximumWidth(laiParametersGroup->width()/5);
    LAIcurveBValue->setValidator(firstValidator);

    QLabel * maxKc= new QLabel(tr("kc max [-]: "));
    maxKcValue = new QLineEdit();
    maxKcValue->setMaximumWidth(laiParametersGroup->width()/5);
    maxKcValue->setValidator(firstValidator);

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
    parametersLaiLayout->addWidget(maxKc, 11, 0);
    parametersLaiLayout->addWidget(maxKcValue, 11, 1);

    QLabel *rootDepthZero = new QLabel(tr("root depth zero [m]: "));
    rootDepthZeroValue = new QLineEdit();
    rootDepthZeroValue->setMaximumWidth(rootParametersGroup->width()/5);
    rootDepthZeroValue->setValidator(firstValidator);

    QLabel *rootDepthMax = new QLabel(tr("root depth max [m]: "));
    rootDepthMaxValue = new QLineEdit();
    rootDepthMaxValue->setMaximumWidth(rootParametersGroup->width()/5);
    rootDepthMaxValue->setValidator(firstValidator);

    QLabel *rootShape = new QLabel(tr("root shape: "));
    rootShapeComboBox = new QComboBox();
    rootShapeComboBox->setMaximumWidth(rootParametersGroup->width()/3);

    for (int i=0; i<numRootDistributionType; i++)
    {
        rootDistributionType type = (rootDistributionType) i;
        rootShapeComboBox->addItem(QString::fromStdString(root::getRootDistributionTypeString(type)));
    }

    QLabel *shapeDeformation = new QLabel(tr("shape deformation [-]: "));
    shapeDeformationValue = new QDoubleSpinBox();
    shapeDeformationValue->setMaximumWidth(rootParametersGroup->width()/5);
    shapeDeformationValue->setMinimum(0);
    shapeDeformationValue->setMaximum(2);
    shapeDeformationValue->setDecimals(1);
    shapeDeformationValue->setSingleStep(0.1);


    degreeDaysInc = new QLabel(tr("degree days root inc [°C]: "));
    degreeDaysIncValue = new QLineEdit();
    degreeDaysIncValue->setMaximumWidth(rootParametersGroup->width()/5);
    degreeDaysIncValue->setValidator(secondValidator);

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
    infoSoilGroup->setLayout(soilInfoLayout);
    laiParametersGroup->setLayout(parametersLaiLayout);
    rootParametersGroup->setLayout(parametersRootDepthLayout);

    infoLayout->addWidget(infoCropGroup);
    infoLayout->addWidget(infoMeteoGroup);
    infoLayout->addWidget(infoSoilGroup);
    infoLayout->addWidget(laiParametersGroup);
    infoLayout->addWidget(rootParametersGroup);

    mainLayout->addLayout(saveButtonLayout);
    mainLayout->addLayout(cropLayout);
    mainLayout->setAlignment(Qt::AlignTop);

    cropLayout->addLayout(infoLayout);
    tabWidget = new QTabWidget;
    tabLAI = new TabLAI();
    tabRootDepth = new TabRootDepth();
    tabRootDensity = new TabRootDensity();
    tabWidget->addTab(tabLAI, tr("LAI development"));
    tabWidget->addTab(tabRootDepth, tr("Root depth"));
    tabWidget->addTab(tabRootDensity, tr("Root density"));
    cropLayout->addWidget(tabWidget);

    this->setLayout(mainLayout);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *fileMenu = new QMenu("File");
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(fileMenu);
    menuBar->addMenu(editMenu);
    this->layout()->setMenuBar(menuBar);

    QAction* openProject = new QAction(tr("&Open CRITERIA-1D Project"), this);
    QAction* openCropDB = new QAction(tr("&Open dbCrop"), this);
    QAction* openMeteoDB = new QAction(tr("&Open dbMeteo"), this);
    QAction* openSoilDB = new QAction(tr("&Open dbSoil"), this);
    saveChanges = new QAction(tr("&Save Changes"), this);

    saveChanges->setEnabled(false);

    QAction* newCrop = new QAction(tr("&New Crop"), this);
    QAction* deleteCrop = new QAction(tr("&Delete Crop"), this);
    restoreData = new QAction(tr("&Restore Data"), this);

    fileMenu->addAction(openProject);
    fileMenu->addAction(openCropDB);
    fileMenu->addAction(openMeteoDB);
    fileMenu->addAction(openSoilDB);
    fileMenu->addAction(saveChanges);

    editMenu->addAction(newCrop);
    editMenu->addAction(deleteCrop);
    editMenu->addAction(restoreData);

    myCrop = nullptr;
    meteoPoint = nullptr;
    cropChanged = false;
    meteoLatBackUp = NODATA;
    layerThickness = 0.02;

    connect(openProject, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenProject);
    connect(openCropDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenCropDB);
    connect(&cropListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseCrop);

    connect(openMeteoDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenMeteoDB);
    connect(&meteoListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseMeteo);
    connect(&yearListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseYear);

    connect(openSoilDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenSoilDB);
    connect(&soilListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseSoil);

    connect(tabWidget, &QTabWidget::currentChanged, [=](int index){ this->tabChanged(index); });

    connect(newCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionNewCrop);
    connect(deleteCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionDeleteCrop);
    connect(restoreData, &QAction::triggered, this, &Crit3DCropWidget::on_actionRestoreData);

    connect(saveButton, &QPushButton::clicked, this, &Crit3DCropWidget::on_actionSave);
    connect(updateButton, &QPushButton::clicked, this, &Crit3DCropWidget::on_actionUpdate);

    //set current tab
    tabChanged(0);
}


void Crit3DCropWidget::on_actionOpenProject()
{
    checkCropUpdate();

    QString projFileName = QFileDialog::getOpenFileName(this, tr("Open Criteria-1D project"), "", tr("Settings files (*.ini)"));

    if (projFileName == "") return;

    QString path = QFileInfo(projFileName).absolutePath()+"/";

    QSettings* projectSettings;
    projectSettings = new QSettings(projFileName, QSettings::IniFormat);
    projectSettings->beginGroup("project");

    path += projectSettings->value("path","").toString();

    QString newDbCropName = projectSettings->value("db_crop","").toString();
    if (newDbCropName.left(1) == ".")
        newDbCropName = QDir::cleanPath(path + newDbCropName);

    QString dbMeteoName = projectSettings->value("db_meteo","").toString();
    if (dbMeteoName.left(1) == ".")
        dbMeteoName = QDir::cleanPath(path + dbMeteoName);

    QString dbSoilName = projectSettings->value("db_soil","").toString();
    if (dbSoilName.left(1) == ".")
        dbSoilName = QDir::cleanPath(path + dbSoilName);

    openCropDB(newDbCropName);
    openMeteoDB(dbMeteoName);
    openSoilDB(dbSoilName);
}


void Crit3DCropWidget::on_actionOpenCropDB()
{
    checkCropUpdate();

    QString newDbCropName = QFileDialog::getOpenFileName(this, tr("Open crop database"), "", tr("SQLite files (*.db)"));

    if (newDbCropName == "")
        return;
    else
        openCropDB(newDbCropName);
}


void Crit3DCropWidget::checkCropUpdate()
{
    if (myCrop != nullptr)
    {
        if (checkIfCropIsChanged())
        {
            QString idCropChanged = QString::fromStdString(myCrop->idCrop);
            QMessageBox::StandardButton confirm;
            QString msg = "Do you want to save changes to crop "+ idCropChanged + " ?";
            confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::Yes);

            if (confirm == QMessageBox::Yes)
            {
                if (updateCrop())
                {
                    if (saveCrop())
                    {
                        // already saved
                        cropChanged = false;
                    }
                }
            }
        }
    }
}


void Crit3DCropWidget::clearCrop()
{
    if (myCrop != nullptr)
    {
        myCrop->clear();
        cropFromDB.clear();
        delete myCrop;
        myCrop = nullptr;
    }
}


void Crit3DCropWidget::openCropDB(QString newDbCropName)
{
    clearCrop();

    QString error;
    if (! openDbCrop(newDbCropName, &dbCrop, &error))
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

    saveChanges->setEnabled(true);
    saveButton->setEnabled(true);
    updateButton->setEnabled(true);
}


void Crit3DCropWidget::on_actionOpenMeteoDB()
{

    QString dbMeteoName = QFileDialog::getOpenFileName(this, tr("Open meteo database"), "", tr("SQLite files (*.db)"));
    if (dbMeteoName == "")
        return;
    else
        openMeteoDB(dbMeteoName);
}


void Crit3DCropWidget::openMeteoDB(QString dbMeteoName)
{
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
    saveChanges->setEnabled(true);
    saveButton->setEnabled(true);
    updateButton->setEnabled(true);
}


void Crit3DCropWidget::on_actionOpenSoilDB()
{
    QString dbSoilName = QFileDialog::getOpenFileName(this, tr("Open soil database"), "", tr("SQLite files (*.db)"));
    if (dbSoilName == "")
        return;
    else
        openSoilDB(dbSoilName);
}


void Crit3DCropWidget::openSoilDB(QString dbSoilName)
{
    QString error;
    if (! openDbSoil(dbSoilName, &dbSoil, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // load default VG parameters
    if (! loadVanGenuchtenParameters(&dbSoil, textureClassList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // load default Driessen parameters
    if (! loadDriessenParameters(&dbSoil, textureClassList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // read soil list
    QStringList soilStringList;
    if (! getSoilList(&dbSoil, &soilStringList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // show soil list
    this->soilListComboBox.clear();
    for (int i = 0; i < soilStringList.size(); i++)
    {
        this->soilListComboBox.addItem(soilStringList[i]);
    }
}


void Crit3DCropWidget::on_actionChooseCrop(QString cropName)
{

    if (cropName.isEmpty())
    {
        return;
    }
    if (checkIfCropIsChanged())
    {
        QString idCropChanged = QString::fromStdString(myCrop->idCrop);
        QMessageBox::StandardButton confirm;
        QString msg = "Do you want to save changes to crop "+ idCropChanged + " ?";
        confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::Yes);

        if (confirm == QMessageBox::Yes)
        {
            if (updateCrop())
            {
                if (saveCrop())
                {
                    cropChanged = false; //already saved
                }
            }
        }
    }

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

    updateCropParam(idCrop);


}

void Crit3DCropWidget::updateCropParam(QString idCrop)
{
    QString error;
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
    LAIminValue->setValue(myCrop->LAImin);
    LAImaxValue->setValue(myCrop->LAImax);
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
    shapeDeformationValue->setValue(myCrop->roots.shapeDeformation);
    rootShapeComboBox->setCurrentText(QString::fromStdString(root::getRootDistributionTypeString(myCrop->roots.rootShape)));
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
        on_actionUpdate();
    }
    cropFromDB = *myCrop;
}


void Crit3DCropWidget::on_actionChooseMeteo(QString idMeteo)
{

    if (idMeteo.isEmpty())
    {
        return;
    }
    QString error, lat, lon;

    if (getLatLonFromIdMeteo(&dbMeteo, idMeteo, &lat, &lon, &error))
    {
        latValue->setValue(lat.toDouble());
        meteoLatBackUp = lat.toDouble();
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
    meteoPoint->latitude = latValue->value();

    // init meteoPoint with 2 years
    int firstYear = year.toInt()-1;
    QDate firstDate(firstYear, 1, 1);
    QDate currentDate(year.toInt(), 1, 1);
    unsigned int numberDays = firstDate.daysInYear() + currentDate.daysInYear();
    meteoPoint->initializeObsDataD(numberDays, getCrit3DDate(firstDate));

    // fill meteoPoint
    if (!fillDailyTempCriteria1D(&dbMeteo, tableMeteo, meteoPoint, QString::number(firstYear), &error))
    {
        QMessageBox::critical(nullptr, "Error!", error + " year: " + QString::number(firstYear));
        return;
    }
    if (!fillDailyTempCriteria1D(&dbMeteo, tableMeteo, meteoPoint, year, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error + " year: " + year);
        return;
    }
    if (myCrop != nullptr)
    {
        on_actionUpdate();
    }

}

void Crit3DCropWidget::on_actionChooseSoil(QString soilCode)
{
    // soilListComboBox has been cleared
    if (soilCode.isEmpty())
    {
        return;
    }

    QString error;
    mySoil.cleanSoil();

    if (! loadSoil(&dbSoil, soilCode, &mySoil, textureClassList, &fittingOptions, &error))
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
    soilLayers = getRegularSoilLayers(&mySoil, layerThickness);
    on_actionUpdate();

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
    QString currentCrop = cropListComboBox.currentText();
    if (checkIfCropIsChanged())
    {
        *myCrop = cropFromDB;
        updateCropParam(QString::fromStdString(myCrop->idCrop));
    }
    latValue->setValue(meteoLatBackUp);
}

void Crit3DCropWidget::on_actionSave()
{

    if (updateCrop())
    {
        if (saveCrop())
        {
            cropChanged = false; //already saved
        }
    }
}


bool Crit3DCropWidget::saveCrop()
{
    QString error;
    if (!updateCropLAIparam(&dbCrop, cropIdValue->text(), myCrop, &error) || !updateCropRootparam(&dbCrop, cropIdValue->text(), myCrop, &error))
    {
        QMessageBox::critical(nullptr, "Update param failed!", error);
        return false;
    }
    cropFromDB = *myCrop;
    return true;
}


void Crit3DCropWidget::on_actionUpdate()
{

    if (!updateCrop() || !updateMeteoPoint())
    {
        //something is null
        return;
    }
    if (!yearListComboBox.currentText().isEmpty())
    {
        updateTabLAI();
        if (!mySoil.code.empty())
        {
            updateTabRootDepth();
            updateTabRootDensity();
        }
    }

}

bool Crit3DCropWidget::updateCrop()
{

    if (myCrop == nullptr)
    {
        return false;
    }
    myCrop->idCrop = cropIdValue->text().toStdString();
    myCrop->type = getCropType(cropTypeValue->text().toStdString());
    if (cropSowing.isVisible())
    {
        myCrop->sowingDoy = cropSowingValue->value();
        myCrop->plantCycle = cropCycleMaxValue->value();
    }
    else
    {
        myCrop->sowingDoy = NODATA;
        myCrop->plantCycle = NODATA;
    }
    myCrop->kcMax = maxKcValue->text().toDouble();
    myCrop->LAImin = LAIminValue->value();
    myCrop->LAImax = LAImaxValue->value();
    if (myCrop->type == FRUIT_TREE)
    {
        myCrop->LAIgrass = LAIgrassValue->text().toDouble();
    }
    else
    {
        myCrop->LAIgrass = NODATA;
    }
    myCrop->thermalThreshold = thermalThresholdValue->text().toDouble();
    myCrop->upperThermalThreshold = upperThermalThresholdValue->text().toDouble();
    myCrop->degreeDaysEmergence = degreeDaysEmergenceValue->text().toDouble();
    myCrop->degreeDaysIncrease = degreeDaysLAIincValue->text().toDouble();
    myCrop->degreeDaysDecrease = degreeDaysLAIdecValue->text().toDouble();
    myCrop->LAIcurve_a = LAIcurveAValue->text().toDouble();
    myCrop->LAIcurve_b = LAIcurveBValue->text().toDouble();

    // root
    myCrop->roots.rootDepthMin = rootDepthZeroValue->text().toDouble();
    myCrop->roots.rootDepthMax = rootDepthMaxValue->text().toDouble();
    myCrop->roots.shapeDeformation = shapeDeformationValue->value();
    myCrop->roots.rootShape = root::getRootDistributionTypeFromString(rootShapeComboBox->currentText().toStdString());
    if (myCrop->isPluriannual())
    {
        myCrop->roots.degreeDaysRootGrowth = NODATA;
    }
    else
    {
        myCrop->roots.degreeDaysRootGrowth = degreeDaysIncValue->text().toDouble();
    }
    cropChanged = true;

    return true;
}

bool Crit3DCropWidget::updateMeteoPoint()
{
    if (meteoPoint == nullptr)
    {
        return false;
    }
    meteoPoint->latitude = latValue->value();
    return true;
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
    if (myCrop != nullptr && meteoPoint != nullptr)
    {
        tabLAI->computeLAI(myCrop, meteoPoint, yearListComboBox.currentText().toInt(), soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDepth()
{
    if (myCrop != nullptr && meteoPoint != nullptr && !mySoil.code.empty())
    {
        tabRootDepth->computeRootDepth(myCrop, meteoPoint, yearListComboBox.currentText().toInt(), soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDensity()
{
    if (myCrop != nullptr && meteoPoint != nullptr && !mySoil.code.empty())
    {
        tabRootDensity->computeRootDensity(myCrop, meteoPoint, yearListComboBox.currentText().toInt(), soilLayers);
    }
}

void Crit3DCropWidget::tabChanged(int index)
{

    if (index == 0) //LAI tab
    {
        rootParametersGroup->hide();
        laiParametersGroup->setVisible(true);
        updateTabLAI();

    }
    else if(index == 1) //root depth tab
    {
        laiParametersGroup->hide();
        rootParametersGroup->setVisible(true);
        if (mySoil.code.empty())
        {
            QString msg = "Open a Db Soil";
            QMessageBox::information(nullptr, "Warning", msg);
            return;
        }
        updateTabRootDepth();
    }
    else if(index == 2) //root density tab
    {
        laiParametersGroup->hide();
        rootParametersGroup->setVisible(true);
        if (mySoil.code.empty())
        {
            QString msg = "Open a Db Soil";
            QMessageBox::information(nullptr, "Warning", msg);
            return;
        }
        updateTabRootDensity();
    }
}

bool Crit3DCropWidget::checkIfCropIsChanged()
{
    // check all editable fields
    if (myCrop == nullptr)
    {
        cropChanged = false;
        return cropChanged;
    }

    if(cropSowingValue->isVisible())
    {
        if (cropFromDB.sowingDoy != cropSowingValue->value() || cropFromDB.plantCycle != cropCycleMaxValue->value())
        {
            cropChanged = true;
            return cropChanged;
        }

    }
    if ( cropFromDB.kcMax != maxKcValue->text().toDouble()
            || cropFromDB.LAImin != LAIminValue->value() || cropFromDB.LAImax != LAImaxValue->value())
    {
        cropChanged = true;
        return cropChanged;

    }
    if (cropFromDB.type == FRUIT_TREE && cropFromDB.LAIgrass != LAIgrassValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    if (cropFromDB.thermalThreshold != thermalThresholdValue->text().toDouble() || cropFromDB.upperThermalThreshold != upperThermalThresholdValue->text().toDouble()
            || cropFromDB.degreeDaysEmergence != degreeDaysEmergenceValue->text().toDouble() || cropFromDB.degreeDaysIncrease != degreeDaysLAIincValue->text().toDouble()
            || cropFromDB.degreeDaysDecrease != degreeDaysLAIdecValue->text().toDouble() || cropFromDB.LAIcurve_a != LAIcurveAValue->text().toDouble() || cropFromDB.LAIcurve_b != LAIcurveBValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    if(cropFromDB.roots.rootDepthMin != rootDepthZeroValue->text().toDouble() || cropFromDB.roots.rootDepthMax != rootDepthMaxValue->text().toDouble()
            || cropFromDB.roots.shapeDeformation != shapeDeformationValue->value() || cropFromDB.roots.rootShape != root::getRootDistributionTypeFromString(rootShapeComboBox->currentText().toStdString()))
    {
        cropChanged = true;
        return cropChanged;
    }
    if (!cropFromDB.isPluriannual() && cropFromDB.roots.degreeDaysRootGrowth != degreeDaysIncValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    else
    {
        cropChanged = false;
    }
    return cropChanged;
}
