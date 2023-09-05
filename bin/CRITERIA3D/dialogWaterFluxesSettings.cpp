#include "dialogWaterFluxesSettings.h"
#include <QtWidgets>
#include <QDoubleValidator>

DialogWaterFluxesSettings::DialogWaterFluxesSettings()
{
    setWindowTitle("3D water fluxes settings");

    // initial water potential [m]
    QGroupBox* initialGroupBox = new QGroupBox("Initial conditions");
    QLabel *initialWaterPotentialLabel = new QLabel(tr("Initial water potential [m]"));
    initialWaterPotentialEdit = new QLineEdit();
    initialWaterPotentialEdit->setFixedWidth(50);
    QDoubleValidator* waterPotentialValidator = new QDoubleValidator(-1000.0, 0.0, 1, initialWaterPotentialEdit);
    waterPotentialValidator->setNotation(QDoubleValidator::StandardNotation);
    initialWaterPotentialEdit->setValidator(waterPotentialValidator);

    QGridLayout *layoutInitial = new QGridLayout();
    layoutInitial->addWidget(initialWaterPotentialLabel, 0 , 0);
    layoutInitial->addWidget(initialWaterPotentialEdit, 0 , 1);
    initialGroupBox->setLayout(layoutInitial);

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

    // ok/cancel buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addWidget(initialGroupBox);
    mainLayout->addWidget(depthGroupBox);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);

    show();
}
