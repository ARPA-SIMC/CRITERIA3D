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

#include "soilWidget.h"
#include "soilDbTools.h"
#include "commonConstants.h"
#include "dialogNewSoil.h"
#include "utilities.h"

#include <math.h>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QString>
#include <QFileDialog>
#include <QLayout>
#include <QMenu>
#include <QLabel>
#include <QMenuBar>
#include <QAction>
#include <QMessageBox>
#include <QFileInfo>
#include <QDebug> //debug


Crit3DSoilWidget::Crit3DSoilWidget()
{
    dbSoilType = DB_SQLITE;
    fittingOptions = new soil::Crit3DFittingOptions();

    this->setWindowTitle(QStringLiteral("Soil"));
    this->resize(1240, 700);

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *nameAndSaveLayout = new QHBoxLayout();
    QHBoxLayout *soilLayout = new QHBoxLayout();
    QVBoxLayout *texturalLayout = new QVBoxLayout();
    QGridLayout *infoLayout = new QGridLayout();

    // check triangle pic and save button pic
    QString docPath, saveButtonPath;
    if (searchDocPath(&docPath))
    {
        picPath = docPath + "img/textural_soil.png";
        saveButtonPath = docPath + "img/saveButton.png";
    }
    else
    {
        picPath = "../img/textural_soil.png";
        saveButtonPath = "../img/saveButton.png";
    }

    pic.load(picPath);
    labelPic = new QLabel();
    labelPic->setPixmap(pic);

    QPixmap pixmap(saveButtonPath);
    QPushButton *saveButton = new QPushButton();
    QIcon ButtonIcon(pixmap);
    saveButton->setIcon(ButtonIcon);
    saveButton->setIconSize(pixmap.rect().size());
    saveButton->setFixedSize(pixmap.rect().size());

    infoGroup = new QGroupBox(tr(""));
    infoGroup->setMaximumWidth(pic.width());
    infoGroup->hide();

    QLabel *soilCodeLabel = new QLabel(tr("Soil code: "));
    soilCodeValue = new QLineEdit();
    soilCodeValue->setReadOnly(true);

    QLabel *satLabel = new QLabel(tr("SAT [m3 m-3]"));
    satValue = new QLineEdit();

    satValue->setReadOnly(true);

    QLabel *fcLabel = new QLabel(tr("FC   [m3 m-3]"));
    fcValue = new QLineEdit();
    fcValue->setReadOnly(true);

    QLabel *wpLabel = new QLabel(tr("WP  [m3 m-3]"));
    wpValue = new QLineEdit();
    wpValue->setReadOnly(true);

    QLabel *awLabel = new QLabel(tr("AW  [m3 m-3]"));
    awValue = new QLineEdit();
    awValue->setReadOnly(true);

    QLabel *potFCLabel = new QLabel(tr("PotFC [kPa]"));
    potFCValue = new QLineEdit();
    potFCValue->setReadOnly(true);

    QLabel *satLegendLabel = new QLabel(tr("SAT = Water content at saturation"));
    QLabel *fcLegendLabel = new QLabel(tr("FC = Water content at Field Capacity"));
    QLabel *wpLegendLabel = new QLabel(tr("WP = Water content at Wilting Point"));
    QLabel *awLegendLabel = new QLabel(tr("AW = Available Water"));
    QLabel *potFCLegendLabel = new QLabel(tr("PotFC = Water Potential at Field Capacity"));

    infoGroup->setTitle(soilName);
    infoLayout->addWidget(soilCodeLabel, 0 , 0);
    infoLayout->addWidget(soilCodeValue, 0 , 1);
    infoLayout->addWidget(satLabel, 1 , 0);
    infoLayout->addWidget(satValue, 1 , 1);
    infoLayout->addWidget(fcLabel, 2 , 0);
    infoLayout->addWidget(fcValue, 2 , 1);
    infoLayout->addWidget(wpLabel, 3 , 0);
    infoLayout->addWidget(wpValue, 3 , 1);
    infoLayout->addWidget(awLabel, 4 , 0);
    infoLayout->addWidget(awValue, 4 , 1);
    infoLayout->addWidget(potFCLabel, 5 , 0);
    infoLayout->addWidget(potFCValue, 5 , 1);

    infoLayout->addWidget(satLegendLabel, 6 , 0);
    infoLayout->addWidget(fcLegendLabel, 7 , 0);
    infoLayout->addWidget(wpLegendLabel, 8 , 0);
    infoLayout->addWidget(awLegendLabel, 9 , 0);
    infoLayout->addWidget(potFCLegendLabel, 10 , 0);
    infoGroup->setLayout(infoLayout);

    int space = 5;
    soilListComboBox.setFixedWidth(pic.size().width() - pixmap.size().width() - space);

    nameAndSaveLayout->setAlignment(Qt::AlignLeft);
    nameAndSaveLayout->addWidget(&soilListComboBox);
    nameAndSaveLayout->addWidget(saveButton);
    mainLayout->addLayout(nameAndSaveLayout);
    mainLayout->addLayout(soilLayout);
    mainLayout->setAlignment(Qt::AlignTop);

    texturalLayout->addWidget(labelPic);
    texturalLayout->setAlignment(Qt::AlignTop);
    texturalLayout->addWidget(infoGroup);

    soilLayout->addLayout(texturalLayout);
    tabWidget = new QTabWidget;
    horizonsTab = new TabHorizons();
    wrDataTab = new TabWaterRetentionData();
    wrCurveTab = new TabWaterRetentionCurve();
    hydraConducCurveTab = new TabHydraulicConductivityCurve();
    tabWidget->addTab(horizonsTab, tr("Horizons"));
    tabWidget->addTab(wrDataTab, tr("Water Retention Data"));
    tabWidget->addTab(wrCurveTab, tr("Water Retention Curve"));
    tabWidget->addTab(hydraConducCurveTab, tr("Hydraulic Conductivity Curve"));

    soilLayout->addWidget(tabWidget);
    this->setLayout(mainLayout);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *fileMenu = new QMenu("File");
    QMenu *editMenu = new QMenu("Edit");
    QMenu *fittingMenu = new QMenu("Fitting");

    menuBar->addMenu(fileMenu);
    menuBar->addMenu(editMenu);
    menuBar->addMenu(fittingMenu);
    this->layout()->setMenuBar(menuBar);

    // actions
    QAction* openSoilDB = new QAction(tr("&Open dbSoil"), this);
    saveChanges = new QAction(tr("&Save Changes"), this);
    QAction* newSoil = new QAction(tr("&New Soil"), this);
    QAction* deleteSoil = new QAction(tr("&Delete Soil"), this);
    restoreData = new QAction(tr("&Restore Data"), this);
    addHorizon = new QAction(tr("&Add Horizon"), this);
    deleteHorizon = new QAction(tr("&Delete Horizon"), this);
    addHorizon->setEnabled(false);
    deleteHorizon->setEnabled(false);
    restoreData->setEnabled(false);
    saveChanges->setEnabled(false);

    useWaterRetentionData = new QAction(tr("&Use Water Retention Data"), this);
    airEntryFixed = new QAction(tr("&Air Entry fixed"), this);
    parameterRestriction = new QAction(tr("&Parameters Restriction (m=1-1/n)"), this);

    useWaterRetentionData->setCheckable(true);
    airEntryFixed->setCheckable(true);
    parameterRestriction->setCheckable(true);
    setFittingMenu();

    connect(openSoilDB, &QAction::triggered, this, &Crit3DSoilWidget::on_actionOpenSoilDB);
    connect(saveChanges, &QAction::triggered, this, &Crit3DSoilWidget::on_actionSave);
    connect(saveButton, &QPushButton::clicked, this, &Crit3DSoilWidget::on_actionSave);
    connect(newSoil, &QAction::triggered, this, &Crit3DSoilWidget::on_actionNewSoil);
    connect(deleteSoil, &QAction::triggered, this, &Crit3DSoilWidget::on_actionDeleteSoil);
    connect(restoreData, &QAction::triggered, this, &Crit3DSoilWidget::on_actionRestoreData);
    connect(addHorizon, &QAction::triggered, this, &Crit3DSoilWidget::on_actionAddHorizon);
    connect(deleteHorizon, &QAction::triggered, this, &Crit3DSoilWidget::on_actionDeleteHorizon);

    connect(useWaterRetentionData, &QAction::triggered, this, &Crit3DSoilWidget::on_actionUseWaterRetentionData);
    connect(airEntryFixed, &QAction::triggered, this, &Crit3DSoilWidget::on_actionAirEntry);
    connect(parameterRestriction, &QAction::triggered, this, &Crit3DSoilWidget::on_actionParameterRestriction);

    connect(&soilListComboBox, &QComboBox::currentTextChanged, this, &Crit3DSoilWidget::on_actionChooseSoil);
    connect(horizonsTab, SIGNAL(horizonSelected(int)), this, SLOT(setInfoTextural(int)));
    connect(wrDataTab, SIGNAL(horizonSelected(int)), this, SLOT(setInfoTextural(int)));
    connect(wrCurveTab, SIGNAL(horizonSelected(int)), this, SLOT(setInfoTextural(int)));
    connect(hydraConducCurveTab, SIGNAL(horizonSelected(int)), this, SLOT(setInfoTextural(int)));

    connect(horizonsTab, SIGNAL(updateSignal()), this, SLOT(updateAll()));
    connect(wrDataTab, SIGNAL(updateSignal()), this, SLOT(updateByTabWR()));
    connect(tabWidget, &QTabWidget::currentChanged, [=](int index){ this->tabChanged(index); });

    fileMenu->addAction(openSoilDB);
    fileMenu->addAction(saveChanges);
    editMenu->addAction(newSoil);
    editMenu->addAction(deleteSoil);
    editMenu->addAction(restoreData);
    editMenu->addAction(addHorizon);
    editMenu->addAction(deleteHorizon);
    fittingMenu->addAction(useWaterRetentionData);
    fittingMenu->addAction(airEntryFixed);
    fittingMenu->addAction(parameterRestriction);

    changed = false;
}


void Crit3DSoilWidget::setFittingMenu()
{
    bool isFittingActive = fittingOptions->useWaterRetentionData;

    useWaterRetentionData->setChecked(isFittingActive);
    airEntryFixed->setEnabled(isFittingActive);
    parameterRestriction->setEnabled(isFittingActive);

    airEntryFixed->setChecked(fittingOptions->airEntryFixed);
    parameterRestriction->setChecked(fittingOptions->mRestriction);
}


void Crit3DSoilWidget::setDbSoil(QString dbSoilName, QString soilCode)
{
    // open soil db
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

    // read soil list
    QStringList soilStringList;
    if (! getSoilList(&dbSoil, &soilStringList, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    // show soil list
    soilListComboBox.clear();
    for (int i = 0; i < soilStringList.size(); i++)
    {
        soilListComboBox.addItem(soilStringList[i]);
    }

    soilListComboBox.setCurrentText(soilCode);
}


void Crit3DSoilWidget::on_actionOpenSoilDB()
{
    QString dbSoilName = QFileDialog::getOpenFileName(this, tr("Open soil database"), "", tr("SQLite files (*.db)"));
    if (dbSoilName == "") return;

    // open soil db
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
    saveChanges->setEnabled(true);
    changed = false;
    wrDataTab->resetHorizonChanged();
}


void Crit3DSoilWidget::cleanInfoGroup()
{
    soilName = QString::fromStdString(mySoil.name);
    satValue->clear();
    fcValue->clear();
    wpValue->clear();
    awValue->clear();
    potFCValue->clear();

    infoGroup->setVisible(true);
    infoGroup->setTitle(soilName);
    soilCodeValue->setText(QString::fromStdString(mySoil.code));
}


void Crit3DSoilWidget::on_actionChooseSoil(QString soilCode)
{
    // re load textural triangle to clean previous circle
    pic.load(picPath);
    labelPic->setPixmap(pic);

    // soilListComboBox has been cleared
    if (soilCode.isEmpty())
    {
        return;
    }

    QString error;
    // somethig has been modified, ask for saving
    if (changed)
    {
        QString soilCodeChanged = QString::fromStdString(mySoil.code);
        QMessageBox::StandardButton confirm;
        QString msg = "Do you want to save changes to soil "+ soilCodeChanged + " ?";
        confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::Yes);

        if (confirm == QMessageBox::Yes)
        {
            if (!updateSoilData(&dbSoil, soilCodeChanged, &mySoil, &error))
            {
                QMessageBox::critical(nullptr, "Error!", error);
                return;
            }

            QVector<int> horizonChanged = wrDataTab->getHorizonChanged();
            // update water_retention DB table
            for (int i = 0; i < horizonChanged.size(); i++)
            {

                if (!updateWaterRetentionData(&dbSoil, soilCodeChanged, &mySoil, horizonChanged[i] + 1, &error))
                {
                    QMessageBox::critical(nullptr, "Error!", error);
                    return;
                }

            }

        }
    }
    changed = false;
    wrDataTab->resetHorizonChanged();

    horizonsTab->resetAll();
    wrDataTab->resetAll();
    wrCurveTab->resetAll();
    hydraConducCurveTab->resetAll();

    mySoil.cleanSoil();

    if (! loadSoil(&dbSoil, soilCode, &mySoil, textureClassList, fittingOptions, &error))
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
    savedSoil = mySoil;
    cleanInfoGroup();
    restoreData->setEnabled(true);

    // circle inside triangle
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        {
            if (soil::getUSDATextureClass(mySoil.horizon[i].dbData.sand, mySoil.horizon[i].dbData.silt, mySoil.horizon[i].dbData.clay) != NODATA)
            {
                // the pic has white space around the triangle: widthTriangle and heightTriangle define triangle size without white space
                double widthOffset = (pic.width() - widthTriangle)/2;
                double heightOffset = (pic.height() - heightTriangle)/2;
                double factor = ( pow ( (pow(100.0, 2.0) - pow(50.0, 2.0)), 0.5) ) / 100;
                // draw new point
                double cx = widthTriangle * ((mySoil.horizon[i].dbData.silt + mySoil.horizon[i].dbData.clay / 2) / 100);
                double cy =  heightTriangle * (1 - mySoil.horizon[i].dbData.clay  / 2 * pow (3, 0.5) / 100 / factor); // tg(60°)=3^0.5
                painter.begin(&pic);
                QPen pen(Qt::red);
                painter.setPen(pen);

                QPointF center(widthOffset + cx, heightOffset + cy);
                painter.setBrush(Qt::transparent);
                painter.drawEllipse(center,4.5,4.5);

                painter.end();
                labelPic->setPixmap(pic);
            }
        }
    }
    tabChanged(tabWidget->currentIndex());
}


void Crit3DSoilWidget::on_actionNewSoil()
{
    if (mySoil.code.empty())
    {
        QString msg = "Open a Db Soil";
        QMessageBox::information(nullptr, "Warning", msg);
        return;
    }
    DialogNewSoil dialog;
    QString error;
    if (dialog.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        int id = dialog.getIdSoilValue();
        QString code = dialog.getCodeSoilValue();
        QString name = dialog.getNameSoilValue();
        QString info = dialog.getInfoSoilValue();
        if (insertSoilData(&dbSoil, id, code, name, info, &error))
        {
            this->soilListComboBox.addItem(code);
            soilListComboBox.setCurrentText(code);
        }
        else
        {
            qDebug() << "Error: " << error;
        }
    }
}


void Crit3DSoilWidget::on_actionDeleteSoil()
{
    QString msg;
    if (soilListComboBox.currentText().isEmpty())
    {
        msg = "Select the soil to be deleted";
        QMessageBox::information(nullptr, "Warning", msg);
    }
    else
    {
        QMessageBox::StandardButton confirm;
        msg = "Are you sure you want to delete "+soilListComboBox.currentText()+" ?";
        confirm = QMessageBox::question(nullptr, "Warning", msg, QMessageBox::Yes|QMessageBox::No, QMessageBox::No);
        QString error;

        if (confirm == QMessageBox::Yes)
        {
            if (deleteSoilData(&dbSoil, soilListComboBox.currentText(), &error))
            {
                soilListComboBox.removeItem(soilListComboBox.currentIndex());
            }
        }
        else
        {
            return;
        }
    }
}


void Crit3DSoilWidget::on_actionUseWaterRetentionData()
{
    fittingOptions->useWaterRetentionData = this->useWaterRetentionData->isChecked();
    setFittingMenu();

    // nothing open
    if (mySoil.code.empty())
    {
        return;
    }
    std::string errorString;
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        soil::setHorizon(&(mySoil.horizon[i]), textureClassList, fittingOptions, &errorString);
    }
    updateAll();
}


void Crit3DSoilWidget::on_actionAirEntry()
{
    fittingOptions->airEntryFixed = this->airEntryFixed->isChecked();

    // nothing open
    if (mySoil.code.empty())
    {
        return;
    }

    std::string errorString;
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        soil::setHorizon(&(mySoil.horizon[i]), textureClassList, fittingOptions, &errorString);
    }
    updateAll();
}


void Crit3DSoilWidget::on_actionParameterRestriction()
{
    fittingOptions->mRestriction = this->parameterRestriction->isChecked();

    // nothing open
    if (mySoil.code.empty())
    {
        return;
    }
    std::string errorString;
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        soil::setHorizon(&(mySoil.horizon[i]), textureClassList, fittingOptions, &errorString);
    }
    updateAll();
}


void Crit3DSoilWidget::on_actionSave()
{
    QString error;
    QString soilCodeChanged = QString::fromStdString(mySoil.code);

    if (!updateSoilData(&dbSoil, soilCodeChanged, &mySoil, &error))
    {
        QMessageBox::critical(nullptr, "Error!", error);
        return;
    }

    QVector<int> horizonChanged = wrDataTab->getHorizonChanged();
    // update water_retention DB table
    for (int i = 0; i < horizonChanged.size(); i++)
    {
        if (!updateWaterRetentionData(&dbSoil, soilCodeChanged, &mySoil, horizonChanged[i]+1, &error))
        {
            QMessageBox::critical(nullptr, "Error!", error);
            return;
        }
    }

    savedSoil = mySoil;
    changed = false;
    wrDataTab->resetHorizonChanged();
}


void Crit3DSoilWidget::on_actionRestoreData()
{
    mySoil = savedSoil;
    horizonsTab->setInsertSoilElement(false);
    wrDataTab->setFillData(false);
    wrCurveTab->setFillElement(false);
    hydraConducCurveTab->setFillElement(false);

    tabChanged(tabWidget->currentIndex());
}

void Crit3DSoilWidget::on_actionAddHorizon()
{
    horizonsTab->addRowClicked();
}

void Crit3DSoilWidget::on_actionDeleteHorizon()
{
    horizonsTab->removeRowClicked();
}

void Crit3DSoilWidget::setInfoTextural(int nHorizon)
{
    // re load textural triangle to clean previous circle
    pic.load(picPath);
    labelPic->setPixmap(pic);
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        if (soil::getUSDATextureClass(mySoil.horizon[i].dbData.sand, mySoil.horizon[i].dbData.silt, mySoil.horizon[i].dbData.clay) != NODATA)
        {
            // the pic has white space around the triangle: widthTriangle and heightTriangle define triangle size without white space
            double widthOffset = (pic.width() - widthTriangle)/2;
            double heightOffset = (pic.height() - heightTriangle)/2;
            double factor = ( pow ( (pow(100.0, 2.0) - pow(50.0, 2.0)), 0.5) ) / 100;
            // draw new point
            double cx = widthTriangle * ((mySoil.horizon[i].dbData.silt + mySoil.horizon[i].dbData.clay / 2) / 100);
            double cy =  heightTriangle * (1 - mySoil.horizon[i].dbData.clay  / 2 * pow (3, 0.5) / 100 / factor); // tg(60°)=3^0.5
            painter.begin(&pic);
            QPen pen(Qt::red);
            painter.setPen(pen);

            QPointF center(widthOffset + cx, heightOffset + cy);
            if (signed(i) == nHorizon)
            {
                painter.setBrush(Qt::red);
                painter.drawEllipse(center,4.5,4.5);
            }
            else
            {
                painter.setBrush(Qt::transparent);
                painter.drawEllipse(center,4.5,4.5);
            }

            painter.end();
            labelPic->setPixmap(pic);
        }
    }

    // nHorizon = -1 : nothing is selected, clear all
    if (nHorizon == -1)
    {
        satValue->setText("");
        fcValue->setText("");
        wpValue->setText("");
        awValue->setText("");
        potFCValue->setText("");
    }
    else
    {
        if (mySoil.horizon[unsigned(nHorizon)].vanGenuchten.thetaS == NODATA)
        {
            satValue->setText(QString::number(NODATA));
        }
        else
        {
            satValue->setText(QString::number(mySoil.horizon[unsigned(nHorizon)].vanGenuchten.thetaS, 'f', 3));
        }

        if (mySoil.horizon[unsigned(nHorizon)].waterContentFC == NODATA)
        {
            fcValue->setText(QString::number(NODATA));
        }
        else
        {
            fcValue->setText(QString::number(mySoil.horizon[unsigned(nHorizon)].waterContentFC, 'f', 3));
        }

        if (mySoil.horizon[unsigned(nHorizon)].waterContentWP == NODATA)
        {
            wpValue->setText(QString::number(NODATA));
        }
        else
        {
            wpValue->setText(QString::number(mySoil.horizon[unsigned(nHorizon)].waterContentWP, 'f', 3));
        }

        if (mySoil.horizon[unsigned(nHorizon)].waterContentFC == NODATA || mySoil.horizon[unsigned(nHorizon)].waterContentWP == NODATA)
        {
            awValue->setText(QString::number(NODATA));
        }
        else
        {
            awValue->setText(QString::number(mySoil.horizon[unsigned(nHorizon)].waterContentFC - mySoil.horizon[unsigned(nHorizon)].waterContentWP, 'f', 3));
        }

        if (mySoil.horizon[unsigned(nHorizon)].fieldCapacity == NODATA)
        {
            potFCValue->setText(QString::number(NODATA));
        }
        else
        {
            potFCValue->setText(QString::number(mySoil.horizon[unsigned(nHorizon)].fieldCapacity, 'f', 3));
        }
    }
}


void Crit3DSoilWidget::tabChanged(int index)
{

    if (soilListComboBox.currentText().isEmpty())
    {
        return;
    }
    if (index == 0)
    {
        if (!horizonsTab->getInsertSoilElement())
        {
            if (mySoil.nrHorizons > 0)
            {
                horizonsTab->insertSoilHorizons(&mySoil, textureClassList, fittingOptions);
                addHorizon->setEnabled(true);
                deleteHorizon->setEnabled(true);
            }
            else
            {
                horizonsTab->resetAll();
                horizonsTab->addRowClicked();
            }

        }
    }
    else if (index == 1) // tab water retention data
    {
        if (!wrDataTab->getFillData())
        {
            if (mySoil.nrHorizons > 0)
            {
                wrDataTab->insertData(&mySoil, textureClassList, fittingOptions);
                addHorizon->setEnabled(false);
                deleteHorizon->setEnabled(false);
            }
            else
            {
                wrDataTab->resetAll();
            }
        }

    }
    else if (index == 2) // tab water retention curve
    {
        if (!wrCurveTab->getFillElement())
        {
            if (mySoil.nrHorizons > 0)
            {
                wrCurveTab->insertElements(&mySoil);
            }
            else
            {
                wrCurveTab->resetAll();
            }
        }

    }
    else if (index == 3) // tab hydraulic conductivity curve
    {
        if (!hydraConducCurveTab->getFillElement())
        {
            if (mySoil.nrHorizons > 0)
            {
                hydraConducCurveTab->insertElements(&mySoil);
            }
            else
            {
                hydraConducCurveTab->resetAll();
            }
        }
    }
}


void Crit3DSoilWidget::updateAll()
{
    changed = true;
    horizonsTab->updateBarHorizon(&mySoil);
    wrDataTab->insertData(&mySoil, textureClassList, fittingOptions);
    wrCurveTab->insertElements(&mySoil);
    hydraConducCurveTab->insertElements(&mySoil);
}

void Crit3DSoilWidget::updateByTabWR()
{
    changed = true;
    wrCurveTab->insertElements(&mySoil);
    horizonsTab->updateTableModel(&mySoil);
}

