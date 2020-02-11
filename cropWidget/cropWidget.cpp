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

    // check save button pic
    QString docPath, saveButtonPath;
    if (searchDocPath(&docPath))
        saveButtonPath = docPath + "img/saveButton.png";
    else
        saveButtonPath = "../img/saveButton.png";

    QPixmap pixmap(saveButtonPath);
    QPushButton *saveButton = new QPushButton();
    QIcon ButtonIcon(pixmap);
    saveButton->setIcon(ButtonIcon);
    saveButton->setIconSize(pixmap.rect().size());
    saveButton->setFixedSize(pixmap.rect().size());

    saveButtonLayout->setAlignment(Qt::AlignLeft);
    saveButtonLayout->addWidget(saveButton);

    QLabel *cropName = new QLabel(tr("CROP_NAME: "));

    QLabel *cropId = new QLabel(tr("ID_CROP: "));
    cropIdValue = new QLineEdit();
    cropIdValue->setReadOnly(true);

    QLabel * cropType= new QLabel(tr("crop type: "));
    cropTypeValue = new QLineEdit();
    cropTypeValue->setReadOnly(true);

    cropSowingValue = new QLineEdit();
    cropSowingValue->setReadOnly(true);

    cropCycleMaxValue = new QLineEdit();
    cropCycleMaxValue->setReadOnly(true);
    cropSowing.setText("sowing DOY: ");
    cropCycleMax.setText("cycle max duration: ");


    infoCropGroup = new QGroupBox(tr(""));
    infoMeteoGroup = new QGroupBox(tr(""));

    infoCropGroup->setFixedWidth(this->width()/4);
    infoMeteoGroup->setFixedWidth(this->width()/4);

    infoCropGroup->setTitle("Crop");
    infoMeteoGroup->setTitle("Meteo");

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
    latValue = new QLineEdit();
    latValue->setReadOnly(true);

    QLabel *lon = new QLabel(tr("longitude: "));
    lonValue = new QLineEdit();
    lonValue->setReadOnly(true);

    meteoInfoLayout->addWidget(meteoName, 0, 0);
    meteoInfoLayout->addWidget(&meteoListComboBox, 0, 1);
    meteoInfoLayout->addWidget(meteoYear, 1, 0);
    meteoInfoLayout->addWidget(&yearListComboBox, 1, 1);
    meteoInfoLayout->addWidget(lat, 2, 0);
    meteoInfoLayout->addWidget(latValue, 2, 1);
    meteoInfoLayout->addWidget(lon, 3, 0);
    meteoInfoLayout->addWidget(lonValue, 3, 1);

    infoCropGroup->setLayout(cropInfoLayout);
    infoMeteoGroup->setLayout(meteoInfoLayout);

    infoLayout->addWidget(infoCropGroup);
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

    myCrop = nullptr;

    connect(openCropDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenCropDB);
    connect(&cropListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseCrop);

    connect(openMeteoDB, &QAction::triggered, this, &Crit3DCropWidget::on_actionOpenMeteoDB);
    connect(&meteoListComboBox, &QComboBox::currentTextChanged, this, &Crit3DCropWidget::on_actionChooseMeteo);
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
    if (! getIdMeteoList(&dbMeteo, &idMeteoList, &error))
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

    if (myCrop->type == HERBACEOUS_ANNUAL ||  myCrop->type == HERBACEOUS_PERENNIAL || myCrop->type == HORTICULTURAL)
    {
        cropSowing.setVisible(true);
        cropCycleMax.setVisible(true);
        cropSowingValue->setText(QString::number(myCrop->sowingDoy));
        cropSowingValue->setVisible(true);
        cropCycleMaxValue->setText(QString::number(myCrop->plantCycle));
        cropCycleMaxValue->setVisible(true);
    }
    else
    {
        cropSowing.setVisible(false);
        cropCycleMax.setVisible(false);
        cropSowingValue->setVisible(false);
        cropCycleMaxValue->setVisible(false);
    }
}

void Crit3DCropWidget::on_actionChooseMeteo(QString idMeteo)
{
    QString error;
    QString lat;
    QString lon;
    if (getLatLonFromIdMeteo(&dbMeteo, idMeteo, &lat, &lon, &error))
    {
        latValue->setText(lat);
        lonValue->setText(lon);
    }
    QString table = getTableNameFromIdMeteo(&dbMeteo, idMeteo, &error);
    QStringList yearList;
    if (!getYears(&dbMeteo, table, &yearList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        this->yearListComboBox.clear();
        return;
    }
    for (int i = 0; i<yearList.size(); i++)
    {
        if ( checkYear(&dbMeteo, table, yearList[i], &error))
        {
            this->yearListComboBox.addItem(yearList[i]);
        }
    }


}

