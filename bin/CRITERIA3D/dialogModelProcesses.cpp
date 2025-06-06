#include "dialogModelProcesses.h"
#include <QtWidgets>

DialogModelProcesses::DialogModelProcesses()
{
    setWindowTitle("3D model processes");

    QGroupBox* processesGroupBox = new QGroupBox("Required");
    QLabel *snowLabel = new QLabel(tr("Snow accumulation and melt"));
    snowProcess = new QCheckBox();
    QLabel *cropLabel = new QLabel(tr("Crop development "));
    cropProcess = new QCheckBox();
    QLabel *waterLabel = new QLabel(tr("Soil water flow "));
    waterFluxesProcess = new QCheckBox();
    QLabel *hydrallLabel = new QLabel(tr("Hydrall model "));
    hydrallProcess = new QCheckBox();
    QLabel *rothCLabel = new QLabel (tr("RothC model"));
    rothCProcess = new QCheckBox();

    QHBoxLayout *layoutProcesses = new QHBoxLayout();
    layoutProcesses->addWidget(waterFluxesProcess);
    layoutProcesses->addWidget(waterLabel);
    layoutProcesses->addWidget(cropProcess);
    layoutProcesses->addWidget(cropLabel);
    layoutProcesses->addWidget(hydrallProcess);
    layoutProcesses->addWidget(hydrallLabel);
    layoutProcesses->addWidget(snowProcess);
    layoutProcesses->addWidget(snowLabel);
    layoutProcesses->addWidget(rothCProcess);
    layoutProcesses->addWidget(rothCLabel);
    processesGroupBox->setLayout(layoutProcesses);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addWidget(processesGroupBox);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);

    show();
}

