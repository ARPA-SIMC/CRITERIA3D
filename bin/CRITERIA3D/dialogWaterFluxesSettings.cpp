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
    QLabel *initialWaterPotentialLabel = new QLabel(tr("Water potential [m]"));
    initialWaterPotentialEdit = new QLineEdit();
    initialWaterPotentialEdit->setFixedWidth(50);
    QDoubleValidator* waterPotentialValidator = new QDoubleValidator(-1000.0, 1.0, 2, initialWaterPotentialEdit);
    waterPotentialValidator->setNotation(QDoubleValidator::StandardNotation);
    initialWaterPotentialEdit->setValidator(waterPotentialValidator);

    QLabel *initialDegreeLabel = new QLabel(tr("Degree of saturation [-]"));
    initialDegreeOfSaturationEdit = new QLineEdit();
    initialDegreeOfSaturationEdit->setFixedWidth(50);
    QDoubleValidator* degreeSatValidator = new QDoubleValidator(0.0, 1.0, 3, initialDegreeOfSaturationEdit);
    degreeSatValidator->setNotation(QDoubleValidator::StandardNotation);
    initialDegreeOfSaturationEdit->setValidator(degreeSatValidator);

    QGridLayout *layoutInitialConditions = new QGridLayout();
    layoutInitialConditions->addWidget(initialWaterPotentialLabel, 0, 0);
    layoutInitialConditions->addWidget(initialWaterPotentialEdit, 0, 1);
    layoutInitialConditions->addWidget(initialDegreeLabel, 1, 0);
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
    QGroupBox* soilGroupBox = new QGroupBox("Soil properties");
    QVBoxLayout *soilLayout = new QVBoxLayout;
    useWaterRetentionFitting = new QRadioButton("Use water retention data");
    soilLayout->addWidget(useWaterRetentionFitting);
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

