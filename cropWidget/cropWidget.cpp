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
#include <QFileInfo>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QLabel>

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

    // check save button pic
    QString saveButtonPath = "../../DOC/img/saveButton.png";
    QFileInfo savePath(saveButtonPath);
    if (! savePath.exists())
    {
        saveButtonPath = "../img/saveButton.png";
    }

    QPixmap pixmap(saveButtonPath);
    QPushButton *saveButton = new QPushButton();
    QIcon ButtonIcon(pixmap);
    saveButton->setIcon(ButtonIcon);
    saveButton->setIconSize(pixmap.rect().size());
    saveButton->setFixedSize(pixmap.rect().size());

    saveButtonLayout->setAlignment(Qt::AlignLeft);
    saveButtonLayout->addWidget(saveButton);

    QLabel *cropId = new QLabel(tr("ID_CROP: "));
    cropIdValue = new QLineEdit();
    cropIdValue->setReadOnly(true);

    QLabel * cropType= new QLabel(tr("crop type: "));
    cropTypeValue = new QLineEdit();
    cropTypeValue->setReadOnly(true);

    QLabel * cropSowing= new QLabel(tr("sowing DOY: "));
    cropSowingValue = new QLineEdit();
    cropSowingValue->setReadOnly(true);

    QLabel * cropCycleMax= new QLabel(tr("cycle max duration: "));
    cropCycleMaxValue = new QLineEdit();
    cropCycleMaxValue->setReadOnly(true);

    infoCropGroup = new QGroupBox(tr(""));
    infoMeteoGroup = new QGroupBox(tr(""));

    infoCropGroup->setTitle("Crop");
    infoMeteoGroup->setTitle("Meteo");

    cropInfoLayout->addWidget(cropId, 0, 0);
    cropInfoLayout->addWidget(cropIdValue, 0, 1);
    cropInfoLayout->addWidget(cropType, 1, 0);
    cropInfoLayout->addWidget(cropTypeValue, 1, 1);
    cropInfoLayout->addWidget(cropSowing, 2, 0);
    cropInfoLayout->addWidget(cropSowingValue, 2, 1);
    cropInfoLayout->addWidget(cropCycleMax, 3, 0);
    cropInfoLayout->addWidget(cropCycleMaxValue, 3, 1);

    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 0);

    infoCropGroup->setLayout(cropInfoLayout);
    infoMeteoGroup->setLayout(meteoInfoLayout);

    infoLayout->addWidget(&cropListComboBox);
    infoLayout->addWidget(infoCropGroup);
    infoLayout->addWidget(&meteoListComboBox);
    infoLayout->addWidget(infoMeteoGroup);

    mainLayout->addLayout(saveButtonLayout);
    mainLayout->addLayout(cropLayout);
    mainLayout->setAlignment(Qt::AlignTop);

    cropLayout->addLayout(infoLayout);
    tabWidget = new QTabWidget;
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
}
