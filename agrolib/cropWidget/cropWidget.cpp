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
#include "cropDbQuery.h"
#include "criteria1DMeteo.h"
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
#include <QSqlQuery>
#include <QSqlError>

#include <QDebug>


Crit3DCropWidget::Crit3DCropWidget()
{
    this->setWindowTitle(QStringLiteral("CRITERIA 1D - Crop Editor"));
    this->resize(1250, 700);

    // font
    QFont myFont = this->font();
    myFont.setPointSize(8);
    this->setFont(myFont);

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *saveButtonLayout = new QHBoxLayout();
    QHBoxLayout *WidgetLayout = new QHBoxLayout();
    QVBoxLayout *infoLayout = new QVBoxLayout();
    QGridLayout *caseInfoLayout = new QGridLayout();
    QGridLayout *cropInfoLayout = new QGridLayout();
    QGridLayout *meteoInfoLayout = new QGridLayout();
    QHBoxLayout *soilInfoLayout = new QHBoxLayout();
    QGridLayout *parametersLaiLayout = new QGridLayout();
    QGridLayout *parametersRootDepthLayout = new QGridLayout();
    QGridLayout *parametersIrrigationLayout = new QGridLayout();
    QGridLayout *parametersWaterStressLayout = new QGridLayout();
    QVBoxLayout *waterContentLayout = new QVBoxLayout();

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

    QPixmap savePixmap;
    QPixmap updatePixmap;
    if (QFileInfo(saveButtonPath).exists())
        savePixmap.load(saveButtonPath);
    else
    {
        QMessageBox::critical(nullptr, "error", "missing file: img/saveButton.png");
    }

    if (QFileInfo(updateButtonPath).exists())
        updatePixmap.load(updateButtonPath);
    else
    {
        QMessageBox::critical(nullptr, "error", "missing file: img/updateButton.png");
    }

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

    QLabel *cropId = new QLabel(tr("ID_CROP: "));
    QLabel *cropName = new QLabel(tr("crop name: "));

    cropNameValue = new QLineEdit();
    cropNameValue->setReadOnly(true);

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

    infoCaseGroup = new QGroupBox(tr(""));
    infoCropGroup = new QGroupBox(tr(""));
    infoMeteoGroup = new QGroupBox(tr(""));
    infoSoilGroup = new QGroupBox(tr(""));
    laiParametersGroup = new QGroupBox(tr(""));
    rootParametersGroup = new QGroupBox(tr(""));
    irrigationParametersGroup = new QGroupBox(tr(""));
    waterStressParametersGroup = new QGroupBox(tr(""));
    waterContentGroup = new QGroupBox(tr(""));

    infoCaseGroup->setFixedWidth(this->width()/5);
    infoCropGroup->setFixedWidth(this->width()/5);
    infoMeteoGroup->setFixedWidth(this->width()/5);
    laiParametersGroup->setFixedWidth(this->width()/5);
    rootParametersGroup->setFixedWidth(this->width()/5);
    irrigationParametersGroup->setFixedWidth(this->width()/5);
    waterStressParametersGroup->setFixedWidth(this->width()/5);
    waterContentGroup->setFixedWidth(this->width()/5);

    infoCaseGroup->setTitle("Case");
    infoCropGroup->setTitle("Crop");
    infoMeteoGroup->setTitle("Meteo");
    infoSoilGroup->setTitle("Soil");
    laiParametersGroup->setTitle("Crop parameters");
    rootParametersGroup->setTitle("root parameters");
    irrigationParametersGroup->setTitle("irrigation parameters");
    waterStressParametersGroup->setTitle("water stress parameters");
    waterContentGroup->setTitle("water content variable");

    caseInfoLayout->addWidget(&caseListComboBox);

    cropInfoLayout->addWidget(cropId, 0, 0);
    cropInfoLayout->addWidget(&cropListComboBox, 0, 1);
    cropInfoLayout->addWidget(cropName, 1, 0);
    cropInfoLayout->addWidget(cropNameValue, 1, 1);
    cropInfoLayout->addWidget(cropType, 2, 0);
    cropInfoLayout->addWidget(cropTypeValue, 2, 1);
    cropInfoLayout->addWidget(&cropSowing, 3, 0);
    cropInfoLayout->addWidget(cropSowingValue, 3, 1);
    cropInfoLayout->addWidget(&cropCycleMax, 4, 0);
    cropInfoLayout->addWidget(cropCycleMaxValue, 4, 1);

    QLabel *meteoName = new QLabel(tr("METEO_NAME: "));

    QLabel *meteoYearFirst = new QLabel(tr("first year: "));
    QLabel *meteoYearLast = new QLabel(tr("last year: "));

    QLabel *lat = new QLabel(tr("latitude: "));
    latValue = new QDoubleSpinBox();
    latValue->setMinimum(-90);
    latValue->setMaximum(90);
    latValue->setDecimals(3);

    meteoInfoLayout->addWidget(meteoName, 0, 0);
    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 1);
    meteoInfoLayout->addWidget(meteoYearFirst, 1, 0);
    meteoInfoLayout->addWidget(&firstYearListComboBox, 1, 1);
    meteoInfoLayout->addWidget(meteoYearLast, 2, 0);
    meteoInfoLayout->addWidget(&lastYearListComboBox, 2, 1);
    meteoInfoLayout->addWidget(lat, 3, 0);
    meteoInfoLayout->addWidget(latValue, 3, 1);

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
    QDoubleValidator* positiveValidator = new QDoubleValidator(0, 99999.0, 3, this);
    firstValidator->setNotation(QDoubleValidator::StandardNotation);
    secondValidator->setNotation(QDoubleValidator::StandardNotation);
    positiveValidator->setNotation(QDoubleValidator::StandardNotation);
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
    degreeDaysEmergenceValue->setValidator(positiveValidator);

    QLabel *degreeDaysLAIinc = new QLabel(tr("degree days phase 1 [°C]: "));
    degreeDaysLAIincValue = new QLineEdit();
    degreeDaysLAIincValue->setMaximumWidth(laiParametersGroup->width()/5);
    degreeDaysLAIincValue->setValidator(positiveValidator);

    QLabel *degreeDaysLAIdec = new QLabel(tr("degree days phase 2 [°C]: "));
    degreeDaysLAIdecValue = new QLineEdit();
    degreeDaysLAIdecValue->setMaximumWidth(laiParametersGroup->width()/5);
    degreeDaysLAIdecValue->setValidator(positiveValidator);

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
    degreeDaysIncValue->setValidator(positiveValidator);

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

    QLabel *irrigationVolume = new QLabel(tr("irrigation quantity [mm]: "));
    irrigationVolumeValue = new QLineEdit();
    irrigationVolumeValue->setText(QString::number(0));
    irrigationVolumeValue->setMaximumWidth(irrigationParametersGroup->width()/5);
    irrigationVolumeValue->setValidator(positiveValidator);
    QLabel *irrigationShift = new QLabel(tr("irrigation shift [days]: "));
    irrigationShiftValue = new QSpinBox();
    irrigationShiftValue->setMaximumWidth(irrigationParametersGroup->width()/5);
    irrigationShiftValue->setMinimum(0);
    irrigationShiftValue->setMaximum(365);
    irrigationShiftValue->setEnabled(false);

    QLabel *degreeDaysStart = new QLabel(tr("degreee days start irrigation [°C]: "));
    degreeDaysStartValue = new QLineEdit();
    degreeDaysStartValue->setMaximumWidth(irrigationParametersGroup->width()/5);
    degreeDaysStartValue->setValidator(positiveValidator);
    degreeDaysStartValue->setEnabled(false);
    QLabel *degreeDaysEnd = new QLabel(tr("degreee days end irrigation [°C]: "));
    degreeDaysEndValue = new QLineEdit();
    degreeDaysEndValue->setMaximumWidth(irrigationParametersGroup->width()/5);
    degreeDaysEndValue->setValidator(positiveValidator);
    degreeDaysEndValue->setEnabled(false);

    parametersIrrigationLayout->addWidget(irrigationVolume, 0, 0);
    parametersIrrigationLayout->addWidget(irrigationVolumeValue, 0, 1);
    parametersIrrigationLayout->addWidget(irrigationShift, 1, 0);
    parametersIrrigationLayout->addWidget(irrigationShiftValue, 1, 1);
    parametersIrrigationLayout->addWidget(degreeDaysStart, 2, 0);
    parametersIrrigationLayout->addWidget(degreeDaysStartValue, 2, 1);
    parametersIrrigationLayout->addWidget(degreeDaysEnd, 3, 0);
    parametersIrrigationLayout->addWidget(degreeDaysEndValue, 3, 1);

    QLabel *psiLeaf = new QLabel(tr("psi leaf [cm]: "));
    psiLeafValue = new QLineEdit();
    psiLeafValue->setMaximumWidth(waterStressParametersGroup->width()/5);
    psiLeafValue->setValidator(positiveValidator);

    QLabel *rawFraction = new QLabel(tr("raw fraction [-]: "));
    rawFractionValue = new QDoubleSpinBox();
    rawFractionValue->setMaximumWidth(waterStressParametersGroup->width()/5);
    rawFractionValue->setMinimum(0);
    rawFractionValue->setMaximum(1);
    rawFractionValue->setDecimals(2);
    rawFractionValue->setSingleStep(0.05);

    QLabel *stressTolerance = new QLabel(tr("stress tolerance [-]: "));
    stressToleranceValue = new QDoubleSpinBox();
    stressToleranceValue->setMaximumWidth(waterStressParametersGroup->width()/5);
    stressToleranceValue->setMinimum(0);
    stressToleranceValue->setMaximum(1);
    stressToleranceValue->setDecimals(2);
    stressToleranceValue->setSingleStep(0.05);

    parametersWaterStressLayout->addWidget(psiLeaf, 0, 0);
    parametersWaterStressLayout->addWidget(psiLeafValue, 0, 1);
    parametersWaterStressLayout->addWidget(rawFraction, 1, 0);
    parametersWaterStressLayout->addWidget(rawFractionValue, 1, 1);
    parametersWaterStressLayout->addWidget(stressTolerance, 2, 0);
    parametersWaterStressLayout->addWidget(stressToleranceValue, 2, 1);

    volWaterContent = new QRadioButton(tr("&volumetric water content [m3 m-3]"));
    degreeSat = new QRadioButton(tr("&degree of saturation [-]"));
    volWaterContent->setChecked(true);
    waterContentLayout->addWidget(volWaterContent);
    waterContentLayout->addWidget(degreeSat);

    infoCaseGroup->setLayout(caseInfoLayout);
    infoCropGroup->setLayout(cropInfoLayout);
    infoMeteoGroup->setLayout(meteoInfoLayout);
    infoSoilGroup->setLayout(soilInfoLayout);
    laiParametersGroup->setLayout(parametersLaiLayout);
    rootParametersGroup->setLayout(parametersRootDepthLayout);
    irrigationParametersGroup->setLayout(parametersIrrigationLayout);
    waterStressParametersGroup->setLayout(parametersWaterStressLayout);
    waterContentGroup->setLayout(waterContentLayout);

    infoLayout->addWidget(infoCaseGroup);
    infoLayout->addWidget(infoCropGroup);
    infoLayout->addWidget(infoMeteoGroup);
    infoLayout->addWidget(infoSoilGroup);
    infoLayout->addWidget(laiParametersGroup);
    infoLayout->addWidget(rootParametersGroup);
    infoLayout->addWidget(irrigationParametersGroup);
    infoLayout->addWidget(waterStressParametersGroup);
    infoLayout->addWidget(waterContentGroup);

    mainLayout->addLayout(saveButtonLayout);
    mainLayout->addLayout(WidgetLayout);
    mainLayout->setAlignment(Qt::AlignTop);

    WidgetLayout->addLayout(infoLayout);
    tabWidget = new QTabWidget;
    tabLAI = new TabLAI();
    tabRootDepth = new TabRootDepth();
    tabRootDensity = new TabRootDensity();
    tabIrrigation = new TabIrrigation();
    tabWaterContent = new TabWaterContent();
    tabWidget->addTab(tabLAI, tr("LAI development"));
    tabWidget->addTab(tabRootDepth, tr("Root depth"));
    tabWidget->addTab(tabRootDensity, tr("Root density"));
    tabWidget->addTab(tabIrrigation, tr("Irrigation"));
    tabWidget->addTab(tabWaterContent, tr("Water Content"));
    WidgetLayout->addWidget(tabWidget);

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
    fileMenu->addSeparator();
    fileMenu->addAction(openCropDB);
    fileMenu->addAction(openMeteoDB);
    fileMenu->addAction(openSoilDB);
    fileMenu->addSeparator();
    fileMenu->addAction(saveChanges);

    editMenu->addAction(newCrop);
    editMenu->addAction(deleteCrop);
    editMenu->addAction(restoreData);

    cropChanged = false;
    meteoLatBackUp = NODATA;

    connect(openProject, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenProject);
    connect(&caseListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseCase);

    connect(openCropDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenCropDB);
    connect(&cropListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseCrop);

    connect(openMeteoDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenMeteoDB);
    connect(&meteoListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseMeteo);
    connect(&firstYearListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseFirstYear);
    connect(&lastYearListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseLastYear);

    connect(openSoilDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenSoilDB);
    connect(&soilListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseSoil);
    connect(irrigationVolumeValue, &QLineEdit::editingFinished, [=](){ this->irrigationVolumeChanged(); });
    connect(volWaterContent, &QRadioButton::toggled, [=](){ this->variableWaterContentChanged(); });

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
    if (dbMeteoName.right(3) == "xml")
        isXmlMeteoGrid = true;
    else
        isXmlMeteoGrid = false;

    QString dbSoilName = projectSettings->value("db_soil","").toString();
    if (dbSoilName.left(1) == ".")
        dbSoilName = QDir::cleanPath(path + dbSoilName);

    QString dbUnitsName = projectSettings->value("db_units","").toString();
    if (dbUnitsName.left(1) == ".")
        dbUnitsName = QDir::cleanPath(path + dbUnitsName);

    this->cropListComboBox.blockSignals(true);
    this->soilListComboBox.blockSignals(true);

    openCropDB(newDbCropName);
    openSoilDB(dbSoilName);

    this->cropListComboBox.blockSignals(false);
    this->soilListComboBox.blockSignals(false);

    this->firstYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.blockSignals(true);

    openMeteoDB(dbMeteoName);

    this->firstYearListComboBox.blockSignals(false);
    this->lastYearListComboBox.blockSignals(false);

    openUnitsDB(dbUnitsName);
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
    if (!myCase.myCrop.idCrop.empty())
    {
        if (checkIfCropIsChanged())
        {
            QString idCropChanged = QString::fromStdString(myCase.myCrop.idCrop);
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


void Crit3DCropWidget::openUnitsDB(QString dbUnitsName)
{  
    QString error;
    if (! loadUnitList(dbUnitsName, unitList, error))
    {
        QMessageBox::critical(nullptr, "Error in DB Units:", error);
        return;
    }

    // unit list
    this->caseListComboBox.blockSignals(true);
    this->caseListComboBox.clear();
    this->caseListComboBox.blockSignals(false);

    for (unsigned int i = 0; i < unitList.size(); i++)
    {
        this->caseListComboBox.addItem(unitList[i].idCase);
    }
}


void Crit3DCropWidget::clearCrop()
{
        myCase.myCrop.clear();
        cropFromDB.clear();
}


void Crit3DCropWidget::openCropDB(QString newDbCropName)
{
    clearCrop();

    QString error;
    if (! openDbCrop(&dbCrop, newDbCropName, &error))
    {
        QMessageBox::critical(nullptr, "Error DB crop", error);
        return;
    }

    // read crop list
    QStringList cropStringList;
    if (! getCropIdList(&dbCrop, &cropStringList, &error))
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

    QString dbMeteoName = QFileDialog::getOpenFileName(this, tr("Open meteo database"), "", tr("SQLite files or XML (*.db *xml)"));
    if (dbMeteoName == "")
        return;
    else
    {
        if (dbMeteoName.right(3) == "xml")
            isXmlMeteoGrid = true;
        else
            isXmlMeteoGrid = false;
        openMeteoDB(dbMeteoName);
    }
}


void Crit3DCropWidget::openMeteoDB(QString dbMeteoName)
{

    QString error;
    QStringList idMeteoList;
    if (isXmlMeteoGrid)
    {
        if (! xmlMeteoGrid.parseXMLGrid(dbMeteoName, &error))
        {
            QMessageBox::critical(nullptr, "Error XML meteo grid", error);
            return;
        }
        if (! xmlMeteoGrid.openDatabase(&error, "observed"))
        {
            QMessageBox::critical(nullptr, "Error DB Grid", error);
            return;
        }
        dbMeteo = xmlMeteoGrid.db();

        if (!xmlMeteoGrid.idDailyList(&error, &idMeteoList))
        {
            QMessageBox::critical(nullptr, "Error daily table list", error);
            return;
        }
    }
    else
    {
        if (! openDbMeteo(dbMeteoName, &dbMeteo, &error))
        {
            QMessageBox::critical(nullptr, "Error DB meteo", error);
            return;
        }

        // read id_meteo list
        if (! getMeteoPointList(&dbMeteo, &idMeteoList, &error))
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }
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


void Crit3DCropWidget::on_actionChooseCase()
{
    this->firstYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.blockSignals(true);

    int index = caseListComboBox.currentIndex();
    QString errorStr;

    // METEO
    QString idMeteo = unitList[index].idMeteo;
    meteoListComboBox.setCurrentText(idMeteo);

    // SOIL
    QString idSoil = getIdSoilString(&dbSoil, unitList[index].idSoilNumber, &errorStr);
    if (idSoil != "")
    {
        soilListComboBox.setCurrentText(idSoil);
    }
    else
    {
        QString soilNumber = QString::number(unitList[index].idSoilNumber);
        QMessageBox::critical(nullptr, "Error!", "Missing soil nr: " + soilNumber + "\n" + errorStr);
    }

    // CROP
    QString idCrop = getCropFromClass(&dbCrop, "crop_class", "id_class", unitList[index].idCropClass, &errorStr);
    if (idCrop != "")
    {
        cropListComboBox.setCurrentText(idCrop);
    }
    else
    {
        QMessageBox::critical(nullptr, "Error!", "Missing crop class: " + unitList[index].idCropClass + "\n" + errorStr);
    }

    this->firstYearListComboBox.blockSignals(false);
    this->lastYearListComboBox.blockSignals(false);
}


void Crit3DCropWidget::on_actionChooseCrop(QString idCrop)
{

    if (idCrop.isEmpty())
    {
        return;
    }
    if (checkIfCropIsChanged())
    {
        QString idCropChanged = QString::fromStdString(myCase.myCrop.idCrop);
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

    // clear previous myCrop
    clearCrop();
    updateCropParam(idCrop);

}


void Crit3DCropWidget::updateCropParam(QString idCrop)
{
    QString error;
    if (!loadCropParameters(&dbCrop, idCrop, &(myCase.myCrop), &error))
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

    cropNameValue->setText(QString::fromStdString(myCase.myCrop.name));
    cropTypeValue->setText(QString::fromStdString(getCropTypeString(myCase.myCrop.type)));

    if (! myCase.myCrop.isPluriannual())
    {
        cropSowing.setVisible(true);
        cropCycleMax.setVisible(true);
        cropSowingValue->setValue(myCase.myCrop.sowingDoy);
        cropSowingValue->setVisible(true);
        cropCycleMaxValue->setValue(myCase.myCrop.plantCycle);
        cropCycleMaxValue->setVisible(true);
    }
    else
    {
        cropSowing.setVisible(false);
        cropCycleMax.setVisible(false);
        cropSowingValue->setVisible(false);
        cropCycleMaxValue->setVisible(false);
    }
    maxKcValue->setText(QString::number(myCase.myCrop.kcMax));

    // LAI parameters
    LAIminValue->setValue(myCase.myCrop.LAImin);
    LAImaxValue->setValue(myCase.myCrop.LAImax);
    if (myCase.myCrop.type == FRUIT_TREE)
    {
        LAIgrass->setVisible(true);
        LAIgrassValue->setVisible(true);
        LAIgrassValue->setText(QString::number(myCase.myCrop.LAIgrass));
    }
    else
    {
        LAIgrass->setVisible(false);
        LAIgrassValue->setVisible(false);
    }
    thermalThresholdValue->setText(QString::number(myCase.myCrop.thermalThreshold));
    upperThermalThresholdValue->setText(QString::number(myCase.myCrop.upperThermalThreshold));
    degreeDaysEmergenceValue->setText(QString::number(myCase.myCrop.degreeDaysEmergence));
    degreeDaysLAIincValue->setText(QString::number(myCase.myCrop.degreeDaysIncrease));
    degreeDaysLAIdecValue->setText(QString::number(myCase.myCrop.degreeDaysDecrease));
    LAIcurveAValue->setText(QString::number(myCase.myCrop.LAIcurve_a));
    LAIcurveBValue->setText(QString::number(myCase.myCrop.LAIcurve_b));

    // root parameters
    rootDepthZeroValue->setText(QString::number(myCase.myCrop.roots.rootDepthMin));
    rootDepthMaxValue->setText(QString::number(myCase.myCrop.roots.rootDepthMax));
    shapeDeformationValue->setValue(myCase.myCrop.roots.shapeDeformation);
    rootShapeComboBox->setCurrentText(QString::fromStdString(root::getRootDistributionTypeString(myCase.myCrop.roots.rootShape)));
    if (myCase.myCrop.isPluriannual())
    {
        degreeDaysInc->setVisible(false);
        degreeDaysIncValue->setVisible(false);
    }
    else
    {
        degreeDaysInc->setVisible(true);
        degreeDaysIncValue->setVisible(true);
        degreeDaysIncValue->setText(QString::number(myCase.myCrop.roots.degreeDaysRootGrowth));
    }
    // irrigation parameters
    irrigationVolumeValue->setText(QString::number(myCase.myCrop.irrigationVolume));
    if (irrigationVolumeValue->text().toDouble() == 0)
    {
        irrigationShiftValue->setValue(0);
        irrigationShiftValue->setEnabled(false);
        degreeDaysStartValue->setText(nullptr);
        degreeDaysStartValue->setEnabled(false);
        degreeDaysEndValue->setText(nullptr);
        degreeDaysEndValue->setEnabled(false);
    }
    else if (irrigationVolumeValue->text().toDouble() > 0)
    {
        irrigationShiftValue->setEnabled(true);
        irrigationShiftValue->setValue(myCase.myCrop.irrigationShift);
        degreeDaysStartValue->setEnabled(true);
        degreeDaysStartValue->setText(QString::number(myCase.myCrop.degreeDaysStartIrrigation));
        degreeDaysEndValue->setEnabled(true);
        degreeDaysEndValue->setText(QString::number(myCase.myCrop.degreeDaysEndIrrigation));
    }
    // water stress parameters
    psiLeafValue->setText(QString::number(myCase.myCrop.psiLeaf));
    rawFractionValue->setValue(myCase.myCrop.fRAW);
    stressToleranceValue->setValue(myCase.myCrop.stressTolerance);

    if (!myCase.meteoPoint.id.empty() && !firstYearListComboBox.currentText().isEmpty())
    {
        on_actionUpdate();
    }
    cropFromDB = myCase.myCrop;
}


void Crit3DCropWidget::on_actionChooseMeteo(QString idMeteo)
{

    if (idMeteo.isEmpty())
    {
        return;
    }
    // clear prev year list
    this->firstYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.blockSignals(true);
    this->firstYearListComboBox.clear();
    this->lastYearListComboBox.clear();
    this->yearList.clear();
    this->firstYearListComboBox.blockSignals(false);

    myCase.meteoPoint.setId(idMeteo.toStdString());
    QString error;

    if (isXmlMeteoGrid)
    {
        if (! xmlMeteoGrid.loadIdMeteoProperties(&error, idMeteo))
        {
            QMessageBox::critical(nullptr, "Error load properties DB Grid", error);
            return;
        }
        double lat;
        if (!xmlMeteoGrid.meteoGrid()->getLatFromId(idMeteo.toStdString(), &lat) )
        {
            error = "Missing observed meteo cell";
            return;
        }
        latValue->setValue(lat);
        meteoLatBackUp = lat;
        tableMeteo = xmlMeteoGrid.tableDaily().prefix + idMeteo + xmlMeteoGrid.tableDaily().postFix;
        if (!xmlMeteoGrid.getYearList(&error, idMeteo, &yearList))
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }
        int pos = 0;
        if (xmlMeteoGrid.gridStructure().isFixedFields())
        {
            QString fieldTmin = xmlMeteoGrid.getDailyVarField(dailyAirTemperatureMin);
            QString fieldTmax = xmlMeteoGrid.getDailyVarField(dailyAirTemperatureMax);
            QString fieldPrec = xmlMeteoGrid.getDailyVarField(dailyPrecipitation);

            // last year can be incomplete
            for (int i = 0; i<yearList.size()-1; i++)
            {

                    if ( !checkYearMeteoGridFixedFields(dbMeteo, tableMeteo, xmlMeteoGrid.tableDaily().fieldTime, fieldTmin, fieldTmax, fieldPrec, yearList[i], &error))
                    {
                        yearList.removeAt(pos);
                        i = i - 1;
                    }
                    else
                    {
                        pos = pos + 1;
                    }
            }
            // store last Date
            getLastDateGrid(dbMeteo, tableMeteo, xmlMeteoGrid.tableDaily().fieldTime, yearList[yearList.size()-1], &lastDBMeteoDate, &error);
        }
        else
        {
            int varCodeTmin = xmlMeteoGrid.getDailyVarCode(dailyAirTemperatureMin);
            int varCodeTmax = xmlMeteoGrid.getDailyVarCode(dailyAirTemperatureMax);
            int varCodePrec = xmlMeteoGrid.getDailyVarCode(dailyPrecipitation);
            if (varCodeTmin == NODATA || varCodeTmax == NODATA || varCodePrec == NODATA)
            {
                error = "Variable not existing";
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }

            // last year can be incomplete
            for (int i = 0; i<yearList.size()-1; i++)
            {

                    if ( !checkYearMeteoGrid(dbMeteo, tableMeteo, xmlMeteoGrid.tableDaily().fieldTime, varCodeTmin, varCodeTmax, varCodePrec, yearList[i], &error))
                    {
                        yearList.removeAt(pos);
                        i = i - 1;
                    }
                    else
                    {
                        pos = pos + 1;
                    }
             }
            // store last Date
            getLastDateGrid(dbMeteo, tableMeteo, xmlMeteoGrid.tableDaily().fieldTime, yearList[yearList.size()-1], &lastDBMeteoDate, &error);
        }
    }
    else
    {
        QString lat,lon;
        if (getLatLonFromIdMeteo(&dbMeteo, idMeteo, &lat, &lon, &error))
        {
            latValue->setValue(lat.toDouble());
            meteoLatBackUp = lat.toDouble();
        }

        tableMeteo = getTableNameFromIdMeteo(&dbMeteo, idMeteo, &error);

        if (!getYearList(&dbMeteo, tableMeteo, &yearList, &error))
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }

        int pos = 0;

        // last year can be incomplete
        for (int i = 0; i<yearList.size()-1; i++)
        {
            if ( !checkYear(&dbMeteo, tableMeteo, yearList[i], &error))
            {
                yearList.removeAt(pos);
                i = i - 1;
            }
            else
            {
                pos = pos + 1;
            }
        }
        // store last Date
        getLastDate(&dbMeteo, tableMeteo, yearList[yearList.size()-1], &lastDBMeteoDate, &error);
    }
    if (yearList.size() == 1)
    {
        onlyOneYear = true;
        yearList.insert(0,QString::number(yearList[0].toInt()-1));
    }
    else
    {
        onlyOneYear = false;
    }

    // add year if exists previous year
    for (int i = 1; i<yearList.size(); i++)
    {
        if (yearList[i].toInt() == yearList[i-1].toInt()+1)
        {
            this->firstYearListComboBox.addItem(yearList[i]);
        }
    }

    this->lastYearListComboBox.blockSignals(false);

}


void Crit3DCropWidget::on_actionChooseFirstYear(QString year)
{

    this->lastYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.clear();
    // add first year
    this->lastYearListComboBox.addItem(year);
    int index = yearList.indexOf(year);

    // add consecutive valid years
    for (int i = index+1; i<yearList.size(); i++)
    {
        if (yearList[i].toInt() == yearList[i-1].toInt()+1)
        {
            this->lastYearListComboBox.addItem(yearList[i]);
        }
        else
        {
            break;
        }
    }
    updateMeteoPointValues();
    this->lastYearListComboBox.blockSignals(false);
}

void Crit3DCropWidget::on_actionChooseLastYear(QString year)
{
    if (year.toInt() - this->firstYearListComboBox.currentText().toInt() > MAX_YEARS)
    {
        QString msg = "Period too long: maximum " + QString::number(MAX_YEARS) + " years";
        QMessageBox::information(nullptr, "Error", msg);
        int max = this->firstYearListComboBox.currentText().toInt() + MAX_YEARS;
        this->lastYearListComboBox.setCurrentText(QString::number(max));
        return;
    }
    updateMeteoPointValues();
}

void Crit3DCropWidget::updateMeteoPointValues()
{

    QString error;

    // clear previous meteoPoint
    myCase.meteoPoint.clear();
    myCase.meteoPoint.id = meteoListComboBox.currentText().toStdString();
    myCase.meteoPoint.latitude = latValue->value();

    // init meteoPoint with all years asked
    int firstYear = this->firstYearListComboBox.currentText().toInt() - 1;
    int lastYear = this->lastYearListComboBox.currentText().toInt();
    QDate firstDate(firstYear, 1, 1);
    QDate lastDate(lastYear, 1, 1);
    QDate myDate = firstDate;
    unsigned int numberDays = 0;
    while (myDate.year() <= lastDate.year())
    {
        numberDays = numberDays + myDate.daysInYear();
        myDate.setDate(myDate.year()+1, 1, 1);
    }
    myCase.meteoPoint.initializeObsDataD(numberDays, getCrit3DDate(firstDate));

    if (isXmlMeteoGrid)
    {
        unsigned row;
        unsigned col;
        if (!xmlMeteoGrid.meteoGrid()->findMeteoPointFromId(&row, &col, myCase.meteoPoint.id) )
        {
            error = "Missing observed meteo cell";
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }

        if (!xmlMeteoGrid.gridStructure().isFixedFields())
        {
            if (!xmlMeteoGrid.loadGridDailyData(&error, QString::fromStdString(myCase.meteoPoint.id), firstDate, QDate(lastDate.year(),12,31)))
            {
                error = "Missing observed data";
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }
        }
        else
        {
            if (!xmlMeteoGrid.loadGridDailyDataFixedFields(&error, QString::fromStdString(myCase.meteoPoint.id), firstDate, QDate(lastDate.year(),12,31)))
            {
                error = "Missing observed data";
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }
        }
        float tmin, tmax, tavg, prec, waterDepth;
        for (int i = 0; i < firstDate.daysTo(QDate(lastDate.year(),12,31))+1; i++)
        {
            Crit3DDate myDate = getCrit3DDate(firstDate.addDays(i));
            tmin = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);

            tmax = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);

            tavg = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
            if (tavg == NODATA)
            {
                tavg = (tmax + tmin)/2;
            }
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);

            prec = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyPrecipitation);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyPrecipitation, prec);

            waterDepth = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyWaterTableDepth);
            myCase.meteoPoint.setMeteoPointValueD(myDate, dailyWaterTableDepth, waterDepth);
        }
        if (onlyOneYear)
        {
            // copy values to prev years
            Crit3DDate myDate = getCrit3DDate(lastDate);
            Crit3DDate prevDate = getCrit3DDate(firstDate);
            for (int i = 0; i < lastDate.daysInYear(); i++)
            {
                prevDate = getCrit3DDate(firstDate).addDays(i);
                myDate = getCrit3DDate(lastDate).addDays(i);
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMin, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMax, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureAvg, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureAvg));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyPrecipitation, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyWaterTableDepth, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
            }
        }
    }
    else
    {
        if (onlyOneYear)
        {
            if (!fillDailyTempPrecCriteria1D(&dbMeteo, tableMeteo, &(myCase.meteoPoint), QString::number(lastYear), &error))
            {
                QMessageBox::critical(nullptr, "Error!", error + " year: " + QString::number(firstYear));
                return;
            }
            // copy values to prev years
            Crit3DDate myDate = getCrit3DDate(lastDate);
            Crit3DDate prevDate = getCrit3DDate(firstDate);
            for (int i = 0; i < lastDate.daysInYear(); i++)
            {
                prevDate = getCrit3DDate(firstDate).addDays(i);
                myDate = getCrit3DDate(lastDate).addDays(i);
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMin, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMax, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureAvg, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureAvg));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyPrecipitation, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
                myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyWaterTableDepth, myCase.meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
            }
        }
        else
        {
            // fill meteoPoint
            for (int year = firstYear; year <= lastYear; year++)
            {
                if (!fillDailyTempPrecCriteria1D(&dbMeteo, tableMeteo, &(myCase.meteoPoint), QString::number(year), &error))
                {
                    QMessageBox::critical(nullptr, "Error!", error + " year: " + QString::number(firstYear));
                    return;
                }
            }
        }
    }

    if (!myCase.myCrop.idCrop.empty())
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
    myCase.mySoil.cleanSoil();

    if (! loadSoil(&dbSoil, soilCode, &(myCase.mySoil), textureClassList, &fittingOptions, &error))
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

    std::string errorString;

    if (! myCase.initializeSoil(errorString))
    {
        QMessageBox::critical(nullptr, "Error!", QString::fromStdString(errorString));
        return;
    }

    if (tabWidget->currentIndex() != 0)
    {
        on_actionUpdate();
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
    QString currentCrop = cropListComboBox.currentText();
    if (checkIfCropIsChanged())
    {
        myCase.myCrop = cropFromDB;
        updateCropParam(QString::fromStdString(myCase.myCrop.idCrop));
    }
    latValue->setValue(meteoLatBackUp);
}

void Crit3DCropWidget::on_actionSave()
{
    QMessageBox::StandardButton confirm;
    QString msg = "Are you sure you want to save "+cropListComboBox.currentText()+" ?";
    confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::No);

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
    else
    {
        return;
    }

}


bool Crit3DCropWidget::saveCrop()
{
    QString error;
    if ( !updateCropLAIparam(&dbCrop, &(myCase.myCrop), &error)
            || !updateCropRootparam(&dbCrop, &(myCase.myCrop), &error)
            || !updateCropIrrigationparam(&dbCrop, &(myCase.myCrop), &error) )
    {
        QMessageBox::critical(nullptr, "Update param failed!", error);
        return false;
    }
    cropFromDB = myCase.myCrop;
    return true;
}


void Crit3DCropWidget::on_actionUpdate()
{

    if (!updateCrop() || !updateMeteoPoint())
    {
        //something is null
        return;
    }
    if (!firstYearListComboBox.currentText().isEmpty())
    {
        if (tabWidget->currentIndex() == 0)
        {
            updateTabLAI();
        }
        else
        {
            if (!myCase.mySoil.code.empty())
            {
                if (tabWidget->currentIndex() == 1)
                {
                    updateTabRootDepth();
                }
                if (tabWidget->currentIndex() == 2)
                {
                    updateTabRootDensity();
                }
                if (tabWidget->currentIndex() == 3)
                {
                    updateTabIrrigation();
                }
                if (tabWidget->currentIndex() == 4)
                {
                    updateTabWaterContent();
                }
            }
        }
    }

}

bool Crit3DCropWidget::updateCrop()
{

    if (myCase.myCrop.idCrop.empty())
    {
        return false;
    }
    myCase.myCrop.type = getCropType(cropTypeValue->text().toStdString());
    if (cropSowing.isVisible())
    {
        myCase.myCrop.sowingDoy = cropSowingValue->value();
        myCase.myCrop.plantCycle = cropCycleMaxValue->value();
    }
    else
    {
        myCase.myCrop.sowingDoy = NODATA;
        myCase.myCrop.plantCycle = NODATA;
    }
    myCase.myCrop.kcMax = maxKcValue->text().toDouble();
    myCase.myCrop.LAImin = LAIminValue->value();
    myCase.myCrop.LAImax = LAImaxValue->value();
    if (myCase.myCrop.type == FRUIT_TREE)
    {
        myCase.myCrop.LAIgrass = LAIgrassValue->text().toDouble();
    }
    else
    {
        myCase.myCrop.LAIgrass = NODATA;
    }
    myCase.myCrop.thermalThreshold = thermalThresholdValue->text().toDouble();
    myCase.myCrop.upperThermalThreshold = upperThermalThresholdValue->text().toDouble();
    myCase.myCrop.degreeDaysEmergence = degreeDaysEmergenceValue->text().toDouble();
    myCase.myCrop.degreeDaysIncrease = degreeDaysLAIincValue->text().toDouble();
    myCase.myCrop.degreeDaysDecrease = degreeDaysLAIdecValue->text().toDouble();
    myCase.myCrop.LAIcurve_a = LAIcurveAValue->text().toDouble();
    myCase.myCrop.LAIcurve_b = LAIcurveBValue->text().toDouble();

    // root
    myCase.myCrop.roots.rootDepthMin = rootDepthZeroValue->text().toDouble();
    myCase.myCrop.roots.rootDepthMax = rootDepthMaxValue->text().toDouble();
    myCase.myCrop.roots.shapeDeformation = shapeDeformationValue->value();
    myCase.myCrop.roots.rootShape = root::getRootDistributionTypeFromString(rootShapeComboBox->currentText().toStdString());
    if (myCase.myCrop.isPluriannual())
    {
        myCase.myCrop.roots.degreeDaysRootGrowth = NODATA;
    }
    else
    {
        myCase.myCrop.roots.degreeDaysRootGrowth = degreeDaysIncValue->text().toDouble();
    }
    // irrigation
    QString error;
    if (irrigationVolumeValue->text().isEmpty())
    {
        error = "irrigation Volume is NULL, insert a valid value";
        QMessageBox::critical(nullptr, "Error irrigation update", error);
        return false;
    }
    else if (irrigationVolumeValue->text().toDouble() == 0)
    {
        myCase.myCrop.irrigationVolume = 0;
        myCase.myCrop.irrigationShift = NODATA;
        myCase.myCrop.degreeDaysStartIrrigation = NODATA;
        myCase.myCrop.degreeDaysEndIrrigation = NODATA;

    }
    else if (irrigationVolumeValue->text().toDouble() > 0)
    {
        if (irrigationShiftValue->value() == 0)
        {
            error = "irrigation shift sould be > 0";
            QMessageBox::critical(nullptr, "Error irrigation update", error);
            return false;
        }
        if (degreeDaysStartValue->text().isEmpty() || degreeDaysEndValue->text().isEmpty())
        {
            error = "irrigation degree days is NULL, insert a valid value";
            QMessageBox::critical(nullptr, "Error irrigation update", error);
            return false;
        }
        myCase.myCrop.irrigationVolume = irrigationVolumeValue->text().toDouble();
        myCase.myCrop.irrigationShift = irrigationShiftValue->value();
        myCase.myCrop.degreeDaysStartIrrigation = degreeDaysStartValue->text().toInt();
        myCase.myCrop.degreeDaysEndIrrigation = degreeDaysEndValue->text().toInt();
    }
    // water stress
    myCase.myCrop.psiLeaf = psiLeafValue->text().toDouble();
    myCase.myCrop.fRAW = rawFractionValue->text().toDouble();
    myCase.myCrop.stressTolerance = stressToleranceValue->text().toDouble();

    cropChanged = true;

    return true;
}

bool Crit3DCropWidget::updateMeteoPoint()
{
    if (myCase.meteoPoint.id.empty())
    {
        return false;
    }
    myCase.meteoPoint.latitude = latValue->value();
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
    if (!myCase.myCrop.idCrop.empty() && !myCase.meteoPoint.id.empty())
    {
        tabLAI->computeLAI(&(myCase.myCrop), &(myCase.meteoPoint), firstYearListComboBox.currentText().toInt(), lastYearListComboBox.currentText().toInt(), lastDBMeteoDate, myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDepth()
{
    if (!myCase.myCrop.idCrop.empty() && !myCase.meteoPoint.id.empty() && !myCase.mySoil.code.empty())
    {
        tabRootDepth->computeRootDepth(&(myCase.myCrop), &(myCase.meteoPoint), firstYearListComboBox.currentText().toInt(), lastYearListComboBox.currentText().toInt(), lastDBMeteoDate, myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDensity()
{
    if (!myCase.myCrop.idCrop.empty() && !myCase.meteoPoint.id.empty() && !myCase.mySoil.code.empty())
    {
        tabRootDensity->computeRootDensity(&(myCase.myCrop), &(myCase.meteoPoint), firstYearListComboBox.currentText().toInt(), lastYearListComboBox.currentText().toInt(), lastDBMeteoDate, myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabIrrigation()
{
    if (!myCase.myCrop.idCrop.empty() && !myCase.meteoPoint.id.empty() && !myCase.mySoil.code.empty())
    {
        tabIrrigation->computeIrrigation(myCase, firstYearListComboBox.currentText().toInt(), lastYearListComboBox.currentText().toInt(), lastDBMeteoDate);
    }
}

void Crit3DCropWidget::updateTabWaterContent()
{
    if (!myCase.myCrop.idCrop.empty() && !myCase.meteoPoint.id.empty() && !myCase.mySoil.code.empty())
    {
        tabWaterContent->computeWaterContent(myCase, firstYearListComboBox.currentText().toInt(), lastYearListComboBox.currentText().toInt(), lastDBMeteoDate, volWaterContent->isChecked());
    }
}

void Crit3DCropWidget::tabChanged(int index)
{

    if (index == 0) //LAI tab
    {
        rootParametersGroup->hide();
        irrigationParametersGroup->hide();
        waterStressParametersGroup->hide();
        waterContentGroup->hide();
        laiParametersGroup->setVisible(true);
        updateTabLAI();

    }
    else if(index == 1) //root depth tab
    {
        laiParametersGroup->hide();
        irrigationParametersGroup->hide();
        waterStressParametersGroup->hide();
        waterContentGroup->hide();
        rootParametersGroup->setVisible(true);
        if (myCase.mySoil.code.empty())
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
        irrigationParametersGroup->hide();
        waterStressParametersGroup->hide();
        waterContentGroup->hide();
        rootParametersGroup->setVisible(true);
        if (myCase.mySoil.code.empty())
        {
            QString msg = "Open a Db Soil";
            QMessageBox::information(nullptr, "Warning", msg);
            return;
        }
        updateTabRootDensity();
    }
    else if(index == 3) //irrigation tab
    {
        laiParametersGroup->hide();
        rootParametersGroup->hide();
        waterContentGroup->hide();
        irrigationParametersGroup->setVisible(true);
        waterStressParametersGroup->setVisible(true);

        if (myCase.mySoil.code.empty())
        {
            QString msg = "Open a Db Soil";
            QMessageBox::information(nullptr, "Warning", msg);
            return;
        }
        updateTabIrrigation();
    }
    else if(index == 4) //water content tab
    {
        laiParametersGroup->hide();
        rootParametersGroup->hide();
        irrigationParametersGroup->hide();
        waterStressParametersGroup->hide();
        waterContentGroup->setVisible(true);

        if (myCase.mySoil.code.empty())
        {
            QString msg = "Open a Db Soil";
            QMessageBox::information(nullptr, "Warning", msg);
            return;
        }
        updateTabWaterContent();
    }

}

bool Crit3DCropWidget::checkIfCropIsChanged()
{
    // check all editable fields
    if (myCase.myCrop.idCrop.empty())
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
    // LAI
    if (cropFromDB.LAImin != LAIminValue->value() || cropFromDB.LAImax != LAImaxValue->value())
    {
        cropChanged = true;
        return cropChanged;

    }
    if (cropFromDB.type == FRUIT_TREE && cropFromDB.LAIgrass != LAIgrassValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    // degree days
    if (cropFromDB.thermalThreshold != thermalThresholdValue->text().toDouble()
            || cropFromDB.upperThermalThreshold != upperThermalThresholdValue->text().toDouble()
            || cropFromDB.degreeDaysEmergence != degreeDaysEmergenceValue->text().toDouble()
            || cropFromDB.degreeDaysIncrease != degreeDaysLAIincValue->text().toDouble()
            || cropFromDB.degreeDaysDecrease != degreeDaysLAIdecValue->text().toDouble()
            || cropFromDB.LAIcurve_a != LAIcurveAValue->text().toDouble()
            || cropFromDB.LAIcurve_b != LAIcurveBValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    // roots
    if(cropFromDB.roots.rootDepthMin != rootDepthZeroValue->text().toDouble()
            || cropFromDB.roots.rootDepthMax != rootDepthMaxValue->text().toDouble()
            || cropFromDB.roots.shapeDeformation != shapeDeformationValue->value()
            || cropFromDB.roots.rootShape != root::getRootDistributionTypeFromString(rootShapeComboBox->currentText().toStdString()))
    {
        cropChanged = true;
        return cropChanged;
    }
    if (!cropFromDB.isPluriannual() && cropFromDB.roots.degreeDaysRootGrowth != degreeDaysIncValue->text().toDouble())
    {
        cropChanged = true;
        return cropChanged;
    }
    // water needs
    if( cropFromDB.kcMax != maxKcValue->text().toDouble()
       || cropFromDB.stressTolerance != stressToleranceValue->text().toDouble()
       || cropFromDB.psiLeaf != psiLeafValue->text().toDouble()
       || cropFromDB.fRAW != rawFractionValue->text().toDouble() )
    {
        cropChanged = true;
        return cropChanged;
    }

    // TODO check irrigation parameters

    cropChanged = false;
    return cropChanged;
}

void Crit3DCropWidget::irrigationVolumeChanged()
{
    if (irrigationVolumeValue->text().toDouble() == 0)
    {
        irrigationShiftValue->setValue(0);
        irrigationShiftValue->setEnabled(false);
        degreeDaysStartValue->setText(nullptr);
        degreeDaysStartValue->setEnabled(false);
        degreeDaysEndValue->setText(nullptr);
        degreeDaysEndValue->setEnabled(false);
    }
    else if (irrigationVolumeValue->text().toDouble() > 0)
    {
        irrigationShiftValue->setEnabled(true);
        irrigationShiftValue->setValue(myCase.myCrop.irrigationShift);
        degreeDaysStartValue->setEnabled(true);
        degreeDaysStartValue->setText(QString::number(myCase.myCrop.degreeDaysStartIrrigation));
        degreeDaysEndValue->setEnabled(true);
        degreeDaysEndValue->setText(QString::number(myCase.myCrop.degreeDaysEndIrrigation));
    }
}

void Crit3DCropWidget::variableWaterContentChanged()
{
    updateTabWaterContent();
}
