#include "dialogWaterFluxesSettings.h"
#include <QtWidgets>
#include <QDoubleValidator>

DialogWaterFluxesSettings::DialogWaterFluxesSettings()
{
    setWindowTitle("Soil water flow settings");

    // model accuracy
    _isUpdateAccuracy = false;
    accuracySlider = new QSlider(Qt::Horizontal);
    accuracySlider->setRange(1, 5);
    accuracySlider->setSingleStep(1);
    accuracySlider->setPageStep(1);
    accuracySlider->setTickInterval(1);
    accuracySlider->setTickPosition(QSlider::TicksBelow);
    updateButton = new QPushButton("  Update accuracy  ");
    QHBoxLayout *accuracyLayout = new QHBoxLayout();
    accuracyLayout->addWidget(accuracySlider);
    accuracyLayout->addWidget(updateButton);
    QGroupBox *accuracyGroupBox = new QGroupBox("Model accuracy");
    accuracyGroupBox->setLayout(accuracyLayout);

    // initial conditions
    useInitialWaterPotential = new QRadioButton("Water potential [m]");
    useInitialDegreeOfSaturation = new QRadioButton("Degree of saturation [-]");

    initialWaterPotentialEdit = new QLineEdit();
    initialWaterPotentialEdit->setFixedWidth(50);
    QDoubleValidator* waterPotentialValidator = new QDoubleValidator(-1000.0, 1.0, 2, initialWaterPotentialEdit);
    waterPotentialValidator->setNotation(QDoubleValidator::StandardNotation);
    initialWaterPotentialEdit->setValidator(waterPotentialValidator);

    initialDegreeOfSaturationEdit = new QLineEdit();
    initialDegreeOfSaturationEdit->setFixedWidth(50);
    QDoubleValidator* degreeSatValidator = new QDoubleValidator(0.0, 1.0, 3, initialDegreeOfSaturationEdit);
    degreeSatValidator->setNotation(QDoubleValidator::StandardNotation);
    initialDegreeOfSaturationEdit->setValidator(degreeSatValidator);

    QGridLayout *layoutInitialConditions = new QGridLayout();
    layoutInitialConditions->addWidget(useInitialWaterPotential, 0, 0);
    layoutInitialConditions->addWidget(initialWaterPotentialEdit, 0, 1);
    layoutInitialConditions->addWidget(useInitialDegreeOfSaturation, 1, 0);
    layoutInitialConditions->addWidget(initialDegreeOfSaturationEdit, 1, 1);

    QGroupBox* initialConditionsGroupBox = new QGroupBox("Initial conditions");
    initialConditionsGroupBox->setLayout(layoutInitialConditions);

    // computation depth [m]
    QGroupBox* depthGroupBox = new QGroupBox("Computation depth");
    onlySurface = new QRadioButton("Only surface");
    allSoilDepth = new QRadioButton("Total soil depth");
    imposedDepth = new QRadioButton("Imposed computation depth [m]");

    imposedComputationDepthEdit = new QLineEdit();
    imposedComputationDepthEdit->setFixedWidth(50);
    QDoubleValidator* imposedDepthValidator = new QDoubleValidator(0.0, 2.0, 2, imposedComputationDepthEdit);
    imposedDepthValidator->setNotation(QDoubleValidator::StandardNotation);
    imposedComputationDepthEdit->setValidator(imposedDepthValidator);

    QGridLayout *layoutDepth = new QGridLayout();
    layoutDepth->addWidget(onlySurface, 0, 0);
    layoutDepth->addWidget(allSoilDepth, 1, 0);
    layoutDepth->addWidget(imposedDepth, 2, 0);
    layoutDepth->addWidget(imposedComputationDepthEdit, 2, 1);
    depthGroupBox->setLayout(layoutDepth);

    // soil properties
    useWaterRetentionFitting = new QRadioButton("Use water retention data");
    QLabel *hvConductivityRatioLabel = new QLabel(tr("Conductivity horizontal/vertical ratio [-]"));
    conductivityHVRatioEdit = new QLineEdit();
    conductivityHVRatioEdit->setFixedWidth(50);
    QDoubleValidator* conductivityRatioValidator = new QDoubleValidator(0.1, 20., 2, conductivityHVRatioEdit);
    conductivityRatioValidator->setNotation(QDoubleValidator::StandardNotation);
    conductivityHVRatioEdit->setValidator(conductivityRatioValidator);

    QGridLayout *soilLayout = new QGridLayout();
    soilLayout->addWidget(hvConductivityRatioLabel, 0, 0);
    soilLayout->addWidget(conductivityHVRatioEdit, 0, 1);
    soilLayout->addWidget(useWaterRetentionFitting, 1, 0);

    QGroupBox* soilGroupBox = new QGroupBox("Soil properties");
    soilGroupBox->setLayout(soilLayout);

    // ok/cancel buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(updateButton, &QPushButton::pressed, this, &DialogWaterFluxesSettings::updateAccuracy);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addWidget(initialConditionsGroupBox);
    mainLayout->addWidget(depthGroupBox);
    mainLayout->addWidget(soilGroupBox);
    mainLayout->addWidget(accuracyGroupBox);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);

    show();
}

