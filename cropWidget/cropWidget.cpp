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
#include "dialogNewProject.h"
#include "cropDbTools.h"
#include "cropDbQuery.h"
#include "criteria1DMeteo.h"
#include "soilDbTools.h"
#include "utilities.h"
#include "commonConstants.h"
#include "soilWidget.h"
#include "meteoWidget.h"
#include "criteria1DMeteo.h"
#include "utilities.h"
#include "basicMath.h"

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


Crit3DCropWidget::Crit3DCropWidget()
{
    setWindowTitle(QStringLiteral("CRITERIA 1D_PRO"));
    resize(1200, 600);

    isRedraw = true;

    // font
    QFont myFont = this->font();
    myFont.setPointSize(8);
    setFont(myFont);

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
        // default appimage linux
        saveButtonPath = QCoreApplication::applicationDirPath() + "/../share/CRITERIA1D/images/saveButton.png";
        updateButtonPath = QCoreApplication::applicationDirPath() + "/../share/CRITERIA1D/images/updateButton.png";
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

    float widthRatio = 0.25;
    infoCaseGroup->setFixedWidth(this->width() * widthRatio);
    infoCropGroup->setFixedWidth(this->width() * widthRatio);
    infoMeteoGroup->setFixedWidth(this->width() * widthRatio);
    laiParametersGroup->setFixedWidth(this->width() * widthRatio);
    rootParametersGroup->setFixedWidth(this->width() * widthRatio);
    irrigationParametersGroup->setFixedWidth(this->width() * widthRatio);
    waterStressParametersGroup->setFixedWidth(this->width() * widthRatio);
    waterContentGroup->setFixedWidth(this->width() * widthRatio);

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

    meteoInfoLayout->addWidget(meteoName, 0, 0);
    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 1);
    meteoInfoLayout->addWidget(meteoYearFirst, 1, 0);
    meteoInfoLayout->addWidget(&firstYearListComboBox, 1, 1);
    meteoInfoLayout->addWidget(meteoYearLast, 2, 0);
    meteoInfoLayout->addWidget(&lastYearListComboBox, 2, 1);

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

    for (int i=0; i < numRootDistributionType; i++)
    {
        rootDistributionType type = rootDistributionType(i);
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
    stressToleranceValue->setSingleStep(0.01);

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
    tabCarbonNitrogen = new TabCarbonNitrogen();

    tabWidget->addTab(tabLAI, tr("LAI development"));
    tabWidget->addTab(tabRootDepth, tr("Root depth"));
    tabWidget->addTab(tabRootDensity, tr("Root density"));
    tabWidget->addTab(tabIrrigation, tr("Irrigation"));
    tabWidget->addTab(tabWaterContent, tr("Water Content"));
    tabWidget->addTab(tabCarbonNitrogen, tr("Carbon Nitrogen"));
    WidgetLayout->addWidget(tabWidget);

    this->setLayout(mainLayout);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *fileMenu = new QMenu("File");
    QMenu *editMenu = new QMenu("Edit");
    viewMenu = new QMenu("View Data");
    viewMenu->setEnabled(false);

    menuBar->addMenu(fileMenu);
    menuBar->addMenu(editMenu);
    menuBar->addMenu(viewMenu);
    this->layout()->setMenuBar(menuBar);

    QAction* openProject = new QAction(tr("&Open Project"), this);
    QAction* newProject = new QAction(tr("&New Project"), this);
    QAction* openCropDB = new QAction(tr("&Open dbCrop"), this);
    QAction* openMeteoDB = new QAction(tr("&Open dbMeteo"), this);
    QAction* openSoilDB = new QAction(tr("&Open dbSoil"), this);

    saveChanges = new QAction(tr("&Save Changes"), this);
    saveChanges->setEnabled(false);
    QAction* executeCase = new QAction(tr("&Execute case"), this);

    QAction* newCrop = new QAction(tr("&New Crop"), this);
    QAction* deleteCrop = new QAction(tr("&Delete Crop"), this);
    restoreData = new QAction(tr("&Restore Data"), this);

    fileMenu->addAction(openProject);
    fileMenu->addAction(newProject);
    fileMenu->addSeparator();
    fileMenu->addAction(openCropDB);
    fileMenu->addAction(openMeteoDB);
    fileMenu->addAction(openSoilDB);
    fileMenu->addSeparator();
    fileMenu->addAction(saveChanges);
    fileMenu->addSeparator();
    fileMenu->addAction(executeCase);

    editMenu->addAction(newCrop);
    editMenu->addAction(deleteCrop);
    editMenu->addAction(restoreData);

    viewWeather = new QAction(tr("&Weather"), this);
    viewSoil = new QAction(tr("&Soil"), this);
    viewMenu->addAction(viewWeather);
    viewMenu->addAction(viewSoil);

    cropChanged = false;

    connect(openProject, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenProject);
    connect(newProject, &QAction::triggered, this, &Crit3DCropWidget::on_actionNewProject);
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

    connect(viewWeather, &QAction::triggered, this, &Crit3DCropWidget::on_actionViewWeather);
    connect(viewSoil, &QAction::triggered, this, &Crit3DCropWidget::on_actionViewSoil);

    connect(newCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionNewCrop);
    connect(deleteCrop, &QAction::triggered, this, &Crit3DCropWidget::on_actionDeleteCrop);
    connect(restoreData, &QAction::triggered, this, &Crit3DCropWidget::on_actionRestoreData);

    connect(saveButton, &QPushButton::clicked, this, &Crit3DCropWidget::on_actionSave);
    connect(updateButton, &QPushButton::clicked, this, &Crit3DCropWidget::on_actionUpdate);

    connect(executeCase, &QAction::triggered, this, &Crit3DCropWidget::on_actionExecuteCase);

    //set current tab
    tabChanged(0);
}


void Crit3DCropWidget::on_actionOpenProject()
{
    isRedraw = false;
    QString dataPath, projectPath;

    if (searchDataPath(&dataPath))
        projectPath = dataPath + PATH_PROJECT;
    else
        projectPath = "";

    checkCropUpdate();
    QString projFileName = QFileDialog::getOpenFileName(this, tr("Open Criteria-1D project"), projectPath, tr("Settings files (*.ini)"));

    if (projFileName == "") return;

    myProject.initialize();
    int myResult = myProject.initializeProject(projFileName);
    if (myResult != CRIT1D_OK)
    {
        QMessageBox::critical(nullptr, "Error", myProject.projectError);
        return;
    }

    this->cropListComboBox.blockSignals(true);
    this->soilListComboBox.blockSignals(true);

    openCropDB(myProject.dbCropName);
    openSoilDB(myProject.dbSoilName);

    this->cropListComboBox.blockSignals(false);
    this->soilListComboBox.blockSignals(false);

    this->firstYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.blockSignals(true);

    openMeteoDB(myProject.dbMeteoName);

    this->firstYearListComboBox.blockSignals(false);
    this->lastYearListComboBox.blockSignals(false);

    openComputationUnitsDB(myProject.dbComputationUnitsName);
    viewMenu->setEnabled(true);
    if (soilListComboBox.count() == 0)
    {
        viewSoil->setEnabled(false);
    }
    else
    {
        viewSoil->setEnabled(true);
    }
    if (meteoListComboBox.count() == 0)
    {
        viewWeather->setEnabled(false);
    }
    else
    {
        viewWeather->setEnabled(true);
    }

    isRedraw = true;
}

void Crit3DCropWidget::on_actionNewProject()
{
    DialogNewProject dialog;
    if (dialog.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        QString dataPath;
        QString projectName = dialog.getProjectName();
        projectName = projectName.simplified().remove(' ');
        if (searchDataPath(&dataPath))
        {
            QString completePath = dataPath+PATH_PROJECT+projectName;
            if(!QDir().mkdir(completePath))
            {
                QMessageBox::StandardButton confirm;
                QString msg = "Project " + completePath + " already exists, do you want to overwrite it?";
                confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::Yes);

                if (confirm == QMessageBox::Yes)
                {
                    clearDir(completePath);
                    QDir().mkdir(completePath+"/data");
                }
                else
                {
                    return;
                }
            }
            else
            {
                QDir().mkdir(completePath+"/data");
            }
            // copy template computational units
            if (!QFile::copy(dataPath + PATH_TEMPLATE + "template_comp_units.db",
                             completePath + "/data/" + "comp_units.db"))
            {
                QMessageBox::critical(nullptr, "Error", "Copy failed: template_comp_units.db");
                return;
            }
            QString db_soil, db_meteo, db_crop;
            // db soil
            if (dialog.getSoilDbOption() == NEW_DB)
            {
                db_soil = "soil.db";
                if (!QFile::copy(dataPath + PATH_TEMPLATE + "template_soil.db", completePath + "/data/" + db_soil))
                {
                    QMessageBox::critical(nullptr, "Error", "Copy failed: template_soil.db");
                    return;
                }
            }
            else if (dialog.getSoilDbOption() == DEFAULT_DB)
            {
                db_soil = "soil_ER_2002.db";
                if (!QFile::copy(dataPath+"SOIL/soil_ER_2002.db", completePath+"/data/"+db_soil))
                {
                    QMessageBox::critical(nullptr, "Error in copy soil_ER_2002.db", "Copy failed");
                    return;
                }
            }
            else if (dialog.getSoilDbOption() == CHOOSE_DB)
            {
                QString soilPath = dialog.getDbSoilCompletePath();
                db_soil = QFileInfo(soilPath).baseName()+".db";
                if (!QFile::copy(soilPath, completePath+"/data/"+db_soil))
                {
                    QMessageBox::critical(nullptr, "Error in copy "+soilPath, "Copy failed");
                    return;
                }
            }
            // db meteo
            if (dialog.getMeteoDbOption() == NEW_DB)
            {
                db_meteo = "meteo.db";
                if (!QFile::copy(dataPath+PATH_TEMPLATE+"template_meteo.db", completePath+"/data/"+db_meteo))
                {
                    QMessageBox::critical(nullptr, "Error in copy template_meteo.db", "Copy failed");
                    return;
                }
            }
            else if (dialog.getMeteoDbOption() == DEFAULT_DB)
            {
                db_meteo = "meteo.db";
                if (!QFile::copy(dataPath+PATH_PROJECT+"test/data/meteo.db", completePath+"/data/"+db_meteo))
                {
                    QMessageBox::critical(nullptr, "Error in copy meteo.db", "Copy failed");
                    return;
                }
            }
            else if (dialog.getMeteoDbOption() == CHOOSE_DB)
            {
                QString meteoPath = dialog.getDbMeteoCompletePath();
                db_meteo = QFileInfo(meteoPath).baseName()+".db";
                if (!QFile::copy(meteoPath, completePath+"/data/"+db_meteo))
                {
                    QMessageBox::critical(nullptr, "Error in copy "+meteoPath, "Copy failed");
                    return;
                }
            }
            // db crop
            if (dialog.getCropDbOption() == DEFAULT_DB)
            {
                db_crop = "crop.db";
                if (!QFile::copy(dataPath+PATH_TEMPLATE+"crop_default.db", completePath+"/data/"+"crop.db"))
                {
                    QMessageBox::critical(nullptr, "Error", "Copy failed: crop_default.db");
                    return;
                }
            }
            else if (dialog.getCropDbOption() == CHOOSE_DB)
            {
                QString cropPath = dialog.getDbCropCompletePath();
                db_crop = QFileInfo(cropPath).baseName()+".db";
                if (!QFile::copy(cropPath, completePath+"/data/"+db_crop))
                {
                    QMessageBox::critical(nullptr, "Error in copy "+cropPath, "Copy failed");
                    return;
                }
            }
            // write .ini
            QSettings* projectSetting = new QSettings(dataPath+PATH_PROJECT+projectName+"/"+projectName+".ini", QSettings::IniFormat);
            projectSetting->beginGroup("software");
                    projectSetting->setValue("software", "CRITERIA1D");
            projectSetting->endGroup();
            projectSetting->beginGroup("project");
                    projectSetting->setValue("path", "./");
                    projectSetting->setValue("name", projectName);
                    projectSetting->setValue("db_soil", "./data/"+db_soil);
                    projectSetting->setValue("db_meteo", "./data/"+db_meteo);
                    projectSetting->setValue("db_crop", "./data/"+db_crop);
                    projectSetting->setValue("db_comp_units", "./data/comp_units.db");
                    projectSetting->setValue("db_output", "./output/"+projectName+".db");
            projectSetting->endGroup();
            projectSetting->sync();

        }

        QMessageBox::information(nullptr, "Success", "project created: " + dataPath+PATH_PROJECT+projectName);
    }
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
    if (!myProject.myCase.crop.idCrop.empty())
    {
        if (checkIfCropIsChanged())
        {
            QString idCropChanged = QString::fromStdString(myProject.myCase.crop.idCrop);
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


void Crit3DCropWidget::openComputationUnitsDB(QString dbComputationUnitsName)
{  
    QString error;
    if (! readComputationUnitList(dbComputationUnitsName, myProject.compUnitList, error))
    {
        QMessageBox::critical(nullptr, "Error in DB Units:", error);
        return;
    }

    // unit list
    this->caseListComboBox.blockSignals(true);
    this->caseListComboBox.clear();
    this->caseListComboBox.blockSignals(false);

    for (unsigned int i = 0; i < myProject.compUnitList.size(); i++)
    {
        this->caseListComboBox.addItem(myProject.compUnitList[i].idCase);
    }
}


void Crit3DCropWidget::clearCrop()
{
        myProject.myCase.crop.clear();
        cropFromDB.clear();
}


void Crit3DCropWidget::openCropDB(QString newDbCropName)
{
    clearCrop();

    QString error;
    if (! openDbCrop(&(myProject.dbCrop), newDbCropName, &error))
    {
        QMessageBox::critical(nullptr, "Error DB crop", error);
        return;
    }

    // read crop list
    QList<QString> cropStringList;
    if (! getCropIdList(&(myProject.dbCrop), &cropStringList, &error))
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
            myProject.isXmlMeteoGrid = true;
        else
            myProject.isXmlMeteoGrid = false;
        openMeteoDB(dbMeteoName);
    }
}


void Crit3DCropWidget::openMeteoDB(QString dbMeteoName)
{

    QString error;
    QList<QString> idMeteoList;
    if (myProject.isXmlMeteoGrid)
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
        myProject.dbMeteo = xmlMeteoGrid.db();

        if (!xmlMeteoGrid.idDailyList(&error, &idMeteoList))
        {
            QMessageBox::critical(nullptr, "Error daily table list", error);
            return;
        }
    }
    else
    {
        if (! openDbMeteo(dbMeteoName, &(myProject.dbMeteo), &error))
        {
            QMessageBox::critical(nullptr, "Error DB meteo", error);
            return;
        }

        // read id_meteo list
        if (! getMeteoPointList(&(myProject.dbMeteo), &idMeteoList, &error))
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
    viewMenu->setEnabled(true);
    if (soilListComboBox.count() == 0)
    {
        viewSoil->setEnabled(false);
    }
    else
    {
        viewSoil->setEnabled(true);
    }
    if (meteoListComboBox.count() == 0)
    {
        viewWeather->setEnabled(false);
    }
    else
    {
        viewWeather->setEnabled(true);
    }

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
    if (! openDbSoil(dbSoilName, &(myProject.dbSoil), &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // load default VG parameters
    if (! loadVanGenuchtenParameters(&(myProject.dbSoil), myProject.soilTexture, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // load default Driessen parameters
    if (! loadDriessenParameters(&(myProject.dbSoil), myProject.soilTexture, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // read soil list
    QList<QString> soilStringList;
    if (! getSoilList(&(myProject.dbSoil), &soilStringList, &error))
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
    viewMenu->setEnabled(true);
    if (soilListComboBox.count() == 0)
    {
        viewSoil->setEnabled(false);
    }
    else
    {
        viewSoil->setEnabled(true);
    }
    if (meteoListComboBox.count() == 0)
    {
        viewWeather->setEnabled(false);
    }
    else
    {
        viewWeather->setEnabled(true);
    }
}


void Crit3DCropWidget::on_actionExecuteCase()
{
    if (! myProject.isProjectLoaded)
    {
        QMessageBox::warning(nullptr, "Warning", "Open a project before.");
        return;
    }

    if (!myProject.computeUnit(myProject.myCase.unit))
    {
        QMessageBox::critical(nullptr, "Error!", myProject.projectError);
    }
    else
    {
        QMessageBox::warning(nullptr, "Case executed: "+ myProject.myCase.unit.idCase, "Output:\n" + QDir().cleanPath(myProject.dbOutputName));
    }
}


void Crit3DCropWidget::on_actionChooseCase()
{
    isRedraw = false;
    this->firstYearListComboBox.blockSignals(true);
    this->lastYearListComboBox.blockSignals(true);

    int index = caseListComboBox.currentIndex();
    QString errorStr;

    myProject.myCase.unit = myProject.compUnitList[unsigned(index)];
    myProject.myCase.fittingOptions.useWaterRetentionData = myProject.myCase.unit.useWaterRetentionData;

    // METEO
    meteoListComboBox.setCurrentText(myProject.myCase.unit.idMeteo);

    // CROP
    myProject.myCase.unit.idCrop = getCropFromClass(&(myProject.dbCrop), "crop_class", "id_class", myProject.myCase.unit.idCropClass, &errorStr);
    if (myProject.myCase.unit.idCrop != "")
    {
        cropListComboBox.setCurrentText(myProject.myCase.unit.idCrop);
        clearCrop();
        updateCropParam(myProject.myCase.unit.idCrop);
    }
    else
    {
        QMessageBox::critical(nullptr, "Error!", "Missing crop class: " + myProject.myCase.unit.idCropClass + "\n" + errorStr);
    }

    // SOIL
    myProject.myCase.unit.idSoil = getIdSoilString(&(myProject.dbSoil), myProject.myCase.unit.idSoilNumber, &errorStr);
    if (myProject.myCase.unit.idSoil != "")
    {
        soilListComboBox.setCurrentText(myProject.myCase.unit.idSoil);
        on_actionChooseSoil(myProject.myCase.unit.idSoil);
    }
    else
    {
        QString soilNumber = QString::number(myProject.myCase.unit.idSoilNumber);
        QMessageBox::critical(nullptr, "Error!", "Missing soil nr: " + soilNumber + "\n" + errorStr);
    }

    this->firstYearListComboBox.blockSignals(false);
    this->lastYearListComboBox.blockSignals(false);

    isRedraw = true;
    on_actionUpdate();
}


void Crit3DCropWidget::on_actionChooseCrop(QString idCrop)
{

    if (idCrop.isEmpty())
    {
        return;
    }
    if (checkIfCropIsChanged())
    {
        QString idCropChanged = QString::fromStdString(myProject.myCase.crop.idCrop);
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

    // clear previous crop
    clearCrop();
    updateCropParam(idCrop);

    if (isRedraw) on_actionUpdate();
}


void Crit3DCropWidget::updateCropParam(QString idCrop)
{
    QString error;
    if (!loadCropParameters(&(myProject.dbCrop), idCrop, &(myProject.myCase.crop), &error))
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

    cropNameValue->setText(QString::fromStdString(myProject.myCase.crop.name));
    cropTypeValue->setText(QString::fromStdString(getCropTypeString(myProject.myCase.crop.type)));

    if (! myProject.myCase.crop.isPluriannual())
    {
        cropSowing.setVisible(true);
        cropCycleMax.setVisible(true);
        cropSowingValue->setValue(myProject.myCase.crop.sowingDoy);
        cropSowingValue->setVisible(true);
        cropCycleMaxValue->setValue(myProject.myCase.crop.plantCycle);
        cropCycleMaxValue->setVisible(true);
    }
    else
    {
        cropSowing.setVisible(false);
        cropCycleMax.setVisible(false);
        cropSowingValue->setVisible(false);
        cropCycleMaxValue->setVisible(false);
    }
    maxKcValue->setText(QString::number(myProject.myCase.crop.kcMax));

    // LAI parameters
    LAIminValue->setValue(myProject.myCase.crop.LAImin);
    LAImaxValue->setValue(myProject.myCase.crop.LAImax);
    if (myProject.myCase.crop.type == FRUIT_TREE)
    {
        LAIgrass->setVisible(true);
        LAIgrassValue->setVisible(true);
        LAIgrassValue->setText(QString::number(myProject.myCase.crop.LAIgrass));
    }
    else
    {
        LAIgrass->setVisible(false);
        LAIgrassValue->setVisible(false);
    }
    thermalThresholdValue->setText(QString::number(myProject.myCase.crop.thermalThreshold));
    upperThermalThresholdValue->setText(QString::number(myProject.myCase.crop.upperThermalThreshold));
    degreeDaysEmergenceValue->setText(QString::number(myProject.myCase.crop.degreeDaysEmergence));
    degreeDaysLAIincValue->setText(QString::number(myProject.myCase.crop.degreeDaysIncrease));
    degreeDaysLAIdecValue->setText(QString::number(myProject.myCase.crop.degreeDaysDecrease));
    LAIcurveAValue->setText(QString::number(myProject.myCase.crop.LAIcurve_a));
    LAIcurveBValue->setText(QString::number(myProject.myCase.crop.LAIcurve_b));

    // root parameters
    rootDepthZeroValue->setText(QString::number(myProject.myCase.crop.roots.rootDepthMin));
    rootDepthMaxValue->setText(QString::number(myProject.myCase.crop.roots.rootDepthMax));
    shapeDeformationValue->setValue(myProject.myCase.crop.roots.shapeDeformation);
    rootShapeComboBox->setCurrentText(QString::fromStdString(root::getRootDistributionTypeString(myProject.myCase.crop.roots.rootShape)));
    if (myProject.myCase.crop.isPluriannual())
    {
        degreeDaysInc->setVisible(false);
        degreeDaysIncValue->setVisible(false);
    }
    else
    {
        degreeDaysInc->setVisible(true);
        degreeDaysIncValue->setVisible(true);
        degreeDaysIncValue->setText(QString::number(myProject.myCase.crop.roots.degreeDaysRootGrowth));
    }
    // irrigation parameters
    irrigationVolumeValue->setText(QString::number(myProject.myCase.crop.irrigationVolume));
    if (irrigationVolumeValue->text() == "0")
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
        irrigationShiftValue->setValue(myProject.myCase.crop.irrigationShift);
        degreeDaysStartValue->setEnabled(true);
        degreeDaysStartValue->setText(QString::number(myProject.myCase.crop.degreeDaysStartIrrigation));
        degreeDaysEndValue->setEnabled(true);
        degreeDaysEndValue->setText(QString::number(myProject.myCase.crop.degreeDaysEndIrrigation));
    }
    // water stress parameters
    psiLeafValue->setText(QString::number(myProject.myCase.crop.psiLeaf));
    rawFractionValue->setValue(myProject.myCase.crop.fRAW);
    stressToleranceValue->setValue(myProject.myCase.crop.stressTolerance);

    cropFromDB = myProject.myCase.crop;
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

    myProject.myCase.meteoPoint.setId(idMeteo.toStdString());
    QString error;

    if (myProject.isXmlMeteoGrid)
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
        myProject.myCase.meteoPoint.latitude = lat;
        meteoTableName = xmlMeteoGrid.tableDaily().prefix + idMeteo + xmlMeteoGrid.tableDaily().postFix;
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

                    if ( !checkYearMeteoGridFixedFields(myProject.dbMeteo, meteoTableName, xmlMeteoGrid.tableDaily().fieldTime, fieldTmin, fieldTmax, fieldPrec, yearList[i], &error))
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
            getLastDateGrid(myProject.dbMeteo, meteoTableName, xmlMeteoGrid.tableDaily().fieldTime, yearList[yearList.size()-1], &(myProject.lastSimulationDate), &error);
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

                    if ( !checkYearMeteoGrid(myProject.dbMeteo, meteoTableName, xmlMeteoGrid.tableDaily().fieldTime, varCodeTmin, varCodeTmax, varCodePrec, yearList[i], &error))
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
            getLastDateGrid(myProject.dbMeteo, meteoTableName, xmlMeteoGrid.tableDaily().fieldTime, yearList[yearList.size()-1], &myProject.lastSimulationDate, &error);
        }
    }
    else
    {
        QString lat,lon;
        if (getLatLonFromIdMeteo(&(myProject.dbMeteo), idMeteo, &lat, &lon, &error))
        {
            myProject.myCase.meteoPoint.latitude = lat.toDouble();
        }

        meteoTableName = getTableNameFromIdMeteo(&(myProject.dbMeteo), idMeteo, &error);

        if (!getYearList(&(myProject.dbMeteo), meteoTableName, &yearList, &error))
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }

        int pos = 0;

        // last year can be incomplete
        for (int i = 0; i<yearList.size()-1; i++)
        {
            if ( !checkYear(&(myProject.dbMeteo), meteoTableName, yearList[i], &error))
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
        getLastDate(&(myProject.dbMeteo), meteoTableName, yearList[yearList.size()-1], &myProject.lastSimulationDate, &error);
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

    // init meteoPoint with all years asked
    int firstYear = this->firstYearListComboBox.currentText().toInt() - 1;
    int lastYear = this->lastYearListComboBox.currentText().toInt();
    QDate firstDate(firstYear, 1, 1);
    QDate lastDate(lastYear, 1, 1);
    QDate myDate = firstDate;
    unsigned int numberDays = 0;
    while (myDate.year() <= lastDate.year())
    {
        numberDays = numberDays + unsigned(myDate.daysInYear());
        myDate.setDate(myDate.year()+1, 1, 1);
    }
    myProject.myCase.meteoPoint.initializeObsDataD(numberDays, getCrit3DDate(firstDate));

    if (myProject.isXmlMeteoGrid)
    {
        unsigned row;
        unsigned col;
        if (!xmlMeteoGrid.meteoGrid()->findMeteoPointFromId(&row, &col, myProject.myCase.meteoPoint.id) )
        {
            error = "Missing observed meteo cell";
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }

        if (!xmlMeteoGrid.gridStructure().isFixedFields())
        {
            if (!xmlMeteoGrid.loadGridDailyData(&error, QString::fromStdString(myProject.myCase.meteoPoint.id), firstDate, QDate(lastDate.year(),12,31)))
            {
                error = "Missing observed data";
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }
        }
        else
        {
            if (!xmlMeteoGrid.loadGridDailyDataFixedFields(&error, QString::fromStdString(myProject.myCase.meteoPoint.id), firstDate, QDate(lastDate.year(),12,31)))
            {
                error = "Missing observed data";
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }
        }
        float tmin, tmax, tavg, prec, waterDepth;
        for (int i = 0; i < (firstDate.daysTo(QDate(lastDate.year(), 12, 31)) + 1); i++)
        {
            Crit3DDate myDate = getCrit3DDate(firstDate.addDays(i));
            tmin = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
            myProject.myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMin, tmin);

            tmax = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
            myProject.myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureMax, tmax);

            tavg = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyAirTemperatureAvg);
            if (isEqual(tavg, NODATA))
            {
                tavg = (tmax + tmin) / 2;
            }
            myProject.myCase.meteoPoint.setMeteoPointValueD(myDate, dailyAirTemperatureAvg, tavg);

            prec = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyPrecipitation);
            myProject.myCase.meteoPoint.setMeteoPointValueD(myDate, dailyPrecipitation, prec);

            waterDepth = xmlMeteoGrid.meteoGrid()->meteoPointPointer(row, col)->getMeteoPointValueD(myDate, dailyWaterTableDepth);
            myProject.myCase.meteoPoint.setMeteoPointValueD(myDate, dailyWaterTableDepth, waterDepth);
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
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMin, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMax, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureAvg, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureAvg));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyPrecipitation, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyWaterTableDepth, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
            }
        }
    }
    else
    {
        if (onlyOneYear)
        {
            if (!fillDailyTempPrecCriteria1D(&(myProject.dbMeteo), meteoTableName, &(myProject.myCase.meteoPoint), QString::number(lastYear), &error))
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
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMin, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMin));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureMax, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureMax));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyAirTemperatureAvg, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyAirTemperatureAvg));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyPrecipitation, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyPrecipitation));
                myProject.myCase.meteoPoint.setMeteoPointValueD(prevDate, dailyWaterTableDepth, myProject.myCase.meteoPoint.getMeteoPointValueD(myDate, dailyWaterTableDepth));
            }
        }
        else
        {
            // fill meteoPoint
            for (int year = firstYear; year <= lastYear; year++)
            {
                if (!fillDailyTempPrecCriteria1D(&(myProject.dbMeteo), meteoTableName,
                                                 &(myProject.myCase.meteoPoint), QString::number(year), &error))
                {
                    QMessageBox::critical(nullptr, "Error!", error + " year: " + QString::number(firstYear));
                    return;
                }
            }
        }
    }

    if (!myProject.myCase.crop.idCrop.empty())
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
    myProject.myCase.mySoil.cleanSoil();

    if (! loadSoil(&(myProject.dbSoil), soilCode, &(myProject.myCase.mySoil),
                  myProject.soilTexture, &(myProject.myCase.fittingOptions), &error))
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

    if (! myProject.myCase.initializeSoil(errorString))
    {
        QMessageBox::critical(nullptr, "Error!", QString::fromStdString(errorString));
        return;
    }

    if (tabWidget->currentIndex() != 0)
    {
        if (isRedraw) on_actionUpdate();
    }
}



void Crit3DCropWidget::on_actionDeleteCrop()
{
    QString msg;
    if (cropListComboBox.currentText().isEmpty())
    {
        msg = "Select the crop to be deleted.";
        QMessageBox::information(nullptr, "Warning", msg);
    }
    else
    {
        QMessageBox::StandardButton confirm;
        msg = "Are you sure you want to delete " + cropListComboBox.currentText() + " ?";
        confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::No);
        QString error;

        if (confirm == QMessageBox::Yes)
        {
            if (deleteCropData(&(myProject.dbCrop), cropListComboBox.currentText(), &error))
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
        myProject.myCase.crop = cropFromDB;
        updateCropParam(QString::fromStdString(myProject.myCase.crop.idCrop));
    }
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
    if ( !updateCropLAIparam(&(myProject.dbCrop), &(myProject.myCase.crop), &error)
            || !updateCropRootparam(&(myProject.dbCrop), &(myProject.myCase.crop), &error)
            || !updateCropIrrigationparam(&(myProject.dbCrop), &(myProject.myCase.crop), &error) )
    {
        QMessageBox::critical(nullptr, "Update param failed!", error);
        return false;
    }
    cropFromDB = myProject.myCase.crop;
    return true;
}


void Crit3DCropWidget::on_actionUpdate()
{
    if (! updateCrop())
    {
        // something is wrong
        return;
    }
    if (! firstYearListComboBox.currentText().isEmpty())
    {
        if (tabWidget->currentIndex() == 0)
        {
            updateTabLAI();
        }
        else
        {
            if ((! myProject.myCase.mySoil.code.empty()) && isRedraw)
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
                if (tabWidget->currentIndex() == 5)
                {
                    updateTabCarbonNitrogen();
                }
            }
        }
    }
}


bool Crit3DCropWidget::updateCrop()
{

    if (myProject.myCase.crop.idCrop.empty())
    {
        return false;
    }
    myProject.myCase.crop.type = getCropType(cropTypeValue->text().toStdString());
    if (cropSowing.isVisible())
    {
        myProject.myCase.crop.sowingDoy = cropSowingValue->value();
        myProject.myCase.crop.plantCycle = cropCycleMaxValue->value();
    }
    else
    {
        myProject.myCase.crop.sowingDoy = NODATA;
        myProject.myCase.crop.plantCycle = NODATA;
    }
    myProject.myCase.crop.kcMax = maxKcValue->text().toDouble();
    myProject.myCase.crop.LAImin = LAIminValue->value();
    myProject.myCase.crop.LAImax = LAImaxValue->value();
    if (myProject.myCase.crop.type == FRUIT_TREE)
    {
        myProject.myCase.crop.LAIgrass = LAIgrassValue->text().toDouble();
    }
    else
    {
        myProject.myCase.crop.LAIgrass = NODATA;
    }
    myProject.myCase.crop.thermalThreshold = thermalThresholdValue->text().toDouble();
    myProject.myCase.crop.upperThermalThreshold = upperThermalThresholdValue->text().toDouble();
    myProject.myCase.crop.degreeDaysEmergence = degreeDaysEmergenceValue->text().toDouble();
    myProject.myCase.crop.degreeDaysIncrease = degreeDaysLAIincValue->text().toDouble();
    myProject.myCase.crop.degreeDaysDecrease = degreeDaysLAIdecValue->text().toDouble();
    myProject.myCase.crop.LAIcurve_a = LAIcurveAValue->text().toDouble();
    myProject.myCase.crop.LAIcurve_b = LAIcurveBValue->text().toDouble();

    // root
    myProject.myCase.crop.roots.rootDepthMin = rootDepthZeroValue->text().toDouble();
    myProject.myCase.crop.roots.rootDepthMax = rootDepthMaxValue->text().toDouble();
    myProject.myCase.crop.roots.shapeDeformation = shapeDeformationValue->value();
    myProject.myCase.crop.roots.rootShape = root::getRootDistributionTypeFromString(rootShapeComboBox->currentText().toStdString());
    if (myProject.myCase.crop.isPluriannual())
    {
        myProject.myCase.crop.roots.degreeDaysRootGrowth = NODATA;
    }
    else
    {
        myProject.myCase.crop.roots.degreeDaysRootGrowth = degreeDaysIncValue->text().toDouble();
    }
    // irrigation
    QString error;
    if (irrigationVolumeValue->text().isEmpty())
    {
        error = "irrigation Volume is NULL, insert a valid value";
        QMessageBox::critical(nullptr, "Error irrigation update", error);
        return false;
    }
    else if (irrigationVolumeValue->text() == "0")
    {
        myProject.myCase.crop.irrigationVolume = 0;
        myProject.myCase.crop.irrigationShift = NODATA;
        myProject.myCase.crop.degreeDaysStartIrrigation = NODATA;
        myProject.myCase.crop.degreeDaysEndIrrigation = NODATA;

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
        myProject.myCase.crop.irrigationVolume = irrigationVolumeValue->text().toDouble();
        myProject.myCase.crop.irrigationShift = irrigationShiftValue->value();
        myProject.myCase.crop.degreeDaysStartIrrigation = degreeDaysStartValue->text().toInt();
        myProject.myCase.crop.degreeDaysEndIrrigation = degreeDaysEndValue->text().toInt();
    }
    // water stress
    myProject.myCase.crop.psiLeaf = psiLeafValue->text().toDouble();
    myProject.myCase.crop.fRAW = rawFractionValue->value();
    myProject.myCase.crop.stressTolerance = stressToleranceValue->value();

    cropChanged = true;

    return true;
}


void Crit3DCropWidget::on_actionNewCrop()
{
    if (!myProject.dbCrop.isOpen())
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
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty())
    {
        tabLAI->computeLAI(&(myProject.myCase.crop), &(myProject.myCase.meteoPoint),
                           firstYearListComboBox.currentText().toInt(),
                           lastYearListComboBox.currentText().toInt(),
                           myProject.lastSimulationDate, myProject.myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDepth()
{
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty() && !myProject.myCase.mySoil.code.empty())
    {
        tabRootDepth->computeRootDepth(&(myProject.myCase.crop), &(myProject.myCase.meteoPoint),
                                       firstYearListComboBox.currentText().toInt(),
                                       lastYearListComboBox.currentText().toInt(),
                                       myProject.lastSimulationDate, myProject.myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabRootDensity()
{
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty() && !myProject.myCase.mySoil.code.empty())
    {
        tabRootDensity->computeRootDensity(&(myProject.myCase.crop), &(myProject.myCase.meteoPoint),
                                           firstYearListComboBox.currentText().toInt(),
                                           lastYearListComboBox.currentText().toInt(),
                                           myProject.lastSimulationDate, myProject.myCase.soilLayers);
    }
}

void Crit3DCropWidget::updateTabIrrigation()
{
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty() && !myProject.myCase.mySoil.code.empty())
    {
        tabIrrigation->computeIrrigation(myProject.myCase, firstYearListComboBox.currentText().toInt(),
                                         lastYearListComboBox.currentText().toInt(),
                                         myProject.lastSimulationDate);
    }
}

void Crit3DCropWidget::updateTabWaterContent()
{
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty() && !myProject.myCase.mySoil.code.empty())
    {
        tabWaterContent->computeWaterContent(myProject.myCase, firstYearListComboBox.currentText().toInt(),
                                             lastYearListComboBox.currentText().toInt(),
                                             myProject.lastSimulationDate, volWaterContent->isChecked());
    }
}

void Crit3DCropWidget::updateTabCarbonNitrogen()
{
    if (!myProject.myCase.crop.idCrop.empty() && !myProject.myCase.meteoPoint.id.empty() && !myProject.myCase.mySoil.code.empty())
    {
        /*tabWaterContent->computeWaterContent(myProject.myCase, firstYearListComboBox.currentText().toInt(),
                                             lastYearListComboBox.currentText().toInt(),
                                             myProject.lastSimulationDate, volWaterContent->isChecked());*/
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
        if (myProject.myCase.mySoil.code.empty())
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
        if (myProject.myCase.mySoil.code.empty())
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

        if (myProject.myCase.mySoil.code.empty())
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

        if (myProject.myCase.mySoil.code.empty())
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
    if (myProject.myCase.crop.idCrop.empty())
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
       || cropFromDB.psiLeaf != psiLeafValue->text().toDouble()
       || ! isEqual(cropFromDB.fRAW, rawFractionValue->value())
       || ! isEqual(cropFromDB.stressTolerance, stressToleranceValue->value()) )
    {
        cropChanged = true;
        return cropChanged;
    }

    // irrigation parameters
    // TODO gestire caso irrigazioni azzerate
    if(irrigationShiftValue->isVisible())
    {
        if( cropFromDB.irrigationVolume != irrigationVolumeValue->text().toDouble()
           || cropFromDB.irrigationShift != irrigationShiftValue->value()
           || cropFromDB.degreeDaysStartIrrigation != degreeDaysStartValue->text().toInt()
           || cropFromDB.degreeDaysEndIrrigation != degreeDaysEndValue->text().toInt() )
        {
            cropChanged = true;
            return cropChanged;
        }
    }

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
        irrigationShiftValue->setValue(cropFromDB.irrigationShift);
        degreeDaysStartValue->setEnabled(true);
        degreeDaysStartValue->setText(QString::number(cropFromDB.degreeDaysStartIrrigation));
        degreeDaysEndValue->setEnabled(true);
        degreeDaysEndValue->setText(QString::number(cropFromDB.degreeDaysEndIrrigation));
    }
}


void Crit3DCropWidget::variableWaterContentChanged()
{
    updateTabWaterContent();
}


bool Crit3DCropWidget::setMeteoSqlite(QString& error)
{

    if (myProject.myCase.meteoPoint.id.empty())
        return false;

    QString queryString = "SELECT * FROM '" + meteoTableName + "' ORDER BY [date]";
    QSqlQuery query = myProject.dbMeteo.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().text() != "")
            error = "dbMeteo error: " + query.lastError().text();
        else
            error = "Missing meteo table:" + meteoTableName;
        return false;
    }

    query.first();
    QDate firstDate = query.value("date").toDate();
    query.last();
    QDate lastDate = query.value("date").toDate();
    unsigned nrDays;
    bool subQuery = false;

    nrDays = unsigned(firstDate.daysTo(lastDate)) + 1;
    if (subQuery)
    {
        query.clear();
        queryString = "SELECT * FROM '" + meteoTableName + "' WHERE date BETWEEN '"
                    + firstDate.toString("yyyy-MM-dd") + "' AND '" + lastDate.toString("yyyy-MM-dd") + "'";
        query = myProject.dbMeteo.exec(queryString);
    }

    // Initialize data
    myProject.myCase.meteoPoint.initializeObsDataD(nrDays, getCrit3DDate(firstDate));

    // Read observed data
    int maxNrDays = NODATA; // all data
    if (! readDailyDataCriteria1D(query, myProject.myCase.meteoPoint, maxNrDays, error))
        return false;

    if (error != "")
        QMessageBox::warning(nullptr, "WARNING!", error);

    return true;

}


void Crit3DCropWidget::on_actionViewWeather()
{
    QString error;
    if (!setMeteoSqlite(error))
    {
        QMessageBox::critical(nullptr, "ERROR!", error);
        return;
    }

    Crit3DMeteoWidget* meteoWidgetPoint = new Crit3DMeteoWidget(myProject.isXmlMeteoGrid, myProject.path, &meteoSettings);

    QDate lastDate = getQDate(myProject.myCase.meteoPoint.getLastDailyData());
    meteoWidgetPoint->setCurrentDate(lastDate);

    meteoWidgetPoint->draw(myProject.myCase.meteoPoint, false);
}


void Crit3DCropWidget::on_actionViewSoil()
{
    Crit3DSoilWidget* soilWidget = new Crit3DSoilWidget();
    soilWidget->setDbSoil(myProject.dbSoil, soilListComboBox.currentText());
}

