#include <QtWidgets>
#include <QList>

#include "project.h"
#include "utilities.h"
#include "aggregation.h"
#include "dialogInterpolation.h"


DialogInterpolation::DialogInterpolation(Project *myProject)
{
    _project = myProject;
    _paramSettings = myProject->parametersSettings;
    _interpolationSettings = &(myProject->interpolationSettings);
    _qualityInterpolationSettings = &(myProject->qualityInterpolationSettings);

    setWindowTitle(tr("Interpolation settings"));

    QHBoxLayout *layoutMain = new QHBoxLayout;
    QVBoxLayout *layoutMainLeft = new QVBoxLayout;
    QVBoxLayout *layoutMainRight = new QVBoxLayout;

    //algorithm
    QGroupBox *groupAlgorithm = new QGroupBox(tr("Algorithm"));
    QHBoxLayout *layoutAlgorithm = new QHBoxLayout;
    QLabel *labelAlgorithm = new QLabel(tr("algorithm"));
    layoutAlgorithm->addWidget(labelAlgorithm);

    std::map<std::string, TInterpolationMethod>::const_iterator itAlg;
    for (itAlg = interpolationMethodNames.begin(); itAlg != interpolationMethodNames.end(); ++itAlg)
        algorithmEdit.addItem(QString::fromStdString(itAlg->first), QString::fromStdString(itAlg->first));

    QString algorithmString = QString::fromStdString(getKeyStringInterpolationMethod(_interpolationSettings->getInterpolationMethod()));
    int indexAlgorithm = algorithmEdit.findData(algorithmString);
    if (indexAlgorithm != -1)
        algorithmEdit.setCurrentIndex(indexAlgorithm);

    layoutAlgorithm->addWidget(&algorithmEdit);
    groupAlgorithm->setLayout(layoutAlgorithm);
    layoutMainLeft->addWidget(groupAlgorithm);

    // grid interpolation
    QGroupBox *groupGrid = new QGroupBox(tr("Gridding"));
    QVBoxLayout *layoutGrid = new QVBoxLayout;

    upscaleFromDemEdit = new QCheckBox(tr("grid upscale from Dem"));
    upscaleFromDemEdit->setChecked(_interpolationSettings->getMeteoGridUpscaleFromDem());
    layoutGrid->addWidget(upscaleFromDemEdit);

    QHBoxLayout * gridAggrLayout = new QHBoxLayout();
    QLabel *labelAggregation = new QLabel(tr("aggregation method"));
    gridAggrLayout->addWidget(labelAggregation);

    std::map<std::string, aggregationMethod>::const_iterator itAggr;
    for (itAggr = aggregationMethodToString.begin(); itAggr != aggregationMethodToString.end(); ++itAggr)
        gridAggregationMethodEdit.addItem(QString::fromStdString(itAggr->first), QString::fromStdString(itAggr->first));

    QString aggregationString = QString::fromStdString(getKeyStringAggregationMethod(_interpolationSettings->getMeteoGridAggrMethod()));
    int indexAggregation = gridAggregationMethodEdit.findData(aggregationString);
    if (indexAggregation != -1)
       gridAggregationMethodEdit.setCurrentIndex(indexAggregation);

    gridAggrLayout->addWidget(&gridAggregationMethodEdit);
    layoutGrid->addLayout(gridAggrLayout);
    groupGrid->setLayout(layoutGrid);
    layoutMainLeft->addWidget(groupGrid);

    connect(upscaleFromDemEdit, SIGNAL(stateChanged(int)), this, SLOT(upscaleFromDemChanged(int)));

    // topographic distances
    QGroupBox *groupTD = new QGroupBox(tr("Topographic distance"));
    QVBoxLayout *layoutTD = new QVBoxLayout();

    topographicDistanceEdit = new QCheckBox(tr("use topographic distance"));
    topographicDistanceEdit->setChecked(_interpolationSettings->getUseTD());
    layoutTD->addWidget(topographicDistanceEdit);

    QHBoxLayout *layoutTdMult = new QHBoxLayout();
    QLabel *labelMaxTd = new QLabel(tr("maximum Td multiplier"));
    QIntValidator *intValTd = new QIntValidator(1, 1000000, this);
    maxTdMultiplierEdit.setFixedWidth(60);
    maxTdMultiplierEdit.setValidator(intValTd);
    maxTdMultiplierEdit.setText(QString::number(_interpolationSettings->getTopoDist_maxKh()));
    layoutTdMult->addWidget(labelMaxTd);
    layoutTdMult->addWidget(&maxTdMultiplierEdit);
    layoutTD->addLayout(layoutTdMult);

    groupTD->setLayout(layoutTD);
    layoutMainLeft->addWidget(groupTD);

    // detrending
    QGroupBox *groupDetrending = new QGroupBox(tr("Detrending"));
    QVBoxLayout *layoutDetrending = new QVBoxLayout();

    QHBoxLayout *layoutR2 = new QHBoxLayout();
    QLabel *labelMinR2 = new QLabel(tr("minimum regression R2"));
    QDoubleValidator *doubleValR2 = new QDoubleValidator(0.0, 1.0, 2, this);
    doubleValR2->setNotation(QDoubleValidator::StandardNotation);
    minRegressionR2Edit.setFixedWidth(30);
    minRegressionR2Edit.setValidator(doubleValR2);
    minRegressionR2Edit.setText(QLocale().toString(_interpolationSettings->getMinRegressionR2()));
    layoutR2->addWidget(labelMinR2);
    layoutR2->addWidget(&minRegressionR2Edit);
    layoutDetrending->addLayout(layoutR2);

    lapseRateCodeEdit = new QCheckBox(tr("use lapse rate code"));
    lapseRateCodeEdit->setChecked(_interpolationSettings->getUseLapseRateCode());
    layoutDetrending->addWidget(lapseRateCodeEdit);

    thermalInversionEdit = new QCheckBox(tr("thermal inversion"));
    thermalInversionEdit->setChecked(_interpolationSettings->getUseThermalInversion());
    layoutDetrending->addWidget(thermalInversionEdit);

    optimalDetrendingEdit = new QCheckBox(tr("optimal detrending"));
    optimalDetrendingEdit->setChecked(_interpolationSettings->getUseBestDetrending());
    layoutDetrending->addWidget(optimalDetrendingEdit);

    connect(optimalDetrendingEdit, SIGNAL(stateChanged(int)), this, SLOT(optimalDetrendingChanged(int)));

    multipleDetrendingEdit = new QCheckBox(tr("multiple detrending"));
    multipleDetrendingEdit->setChecked(_interpolationSettings->getUseMultipleDetrending());
    layoutDetrending->addWidget(multipleDetrendingEdit);

    connect(multipleDetrendingEdit, SIGNAL(stateChanged(int)), this, SLOT(multipleDetrendingChanged(int)));

    localDetrendingEdit = new QCheckBox(tr("local detrending"));
    localDetrendingEdit->setChecked(_interpolationSettings->getUseLocalDetrending());

    connect(localDetrendingEdit, SIGNAL(stateChanged(int)), this, SLOT(localDetrendingChanged(int)));

    doNotRetrendEdit = new QCheckBox(tr("do not retrend"));
    doNotRetrendEdit->setChecked(_interpolationSettings->getUseDoNotRetrend());

    retrendOnlyEdit = new QCheckBox(tr("retrend only"));
    retrendOnlyEdit->setChecked(_interpolationSettings->getUseRetrendOnly());

    QLabel *labelMinPointsLocalDetrendingEdit = new QLabel(tr("minimum points for local detrending"));
    QIntValidator *intValMinPoints = new QIntValidator(1, 1000, this);
    minPointsLocalDetrendingEdit.setFixedWidth(30);
    minPointsLocalDetrendingEdit.setValidator(intValMinPoints);
    minPointsLocalDetrendingEdit.setText(QString::number(_interpolationSettings->getMinPointsLocalDetrending()));
    layoutDetrending->addWidget(labelMinPointsLocalDetrendingEdit);
    layoutDetrending->addWidget(&minPointsLocalDetrendingEdit);

    layoutDetrending->addWidget(localDetrendingEdit);
    layoutDetrending->addWidget(doNotRetrendEdit);
    layoutDetrending->addWidget(retrendOnlyEdit);

    QLabel *labelElFunction = new QLabel(tr("fitting function for elevation"));
    layoutDetrending->addWidget(labelElFunction);

    std::map<std::string, TFittingFunction>::const_iterator itElFunc;
    for (itElFunc = fittingFunctionNames.begin(); itElFunc != fittingFunctionNames.end(); ++itElFunc)
    {
        if (itElFunc->first == "linear")
            continue;
        elevationFunctionEdit.addItem(QString::fromStdString(itElFunc->first), QString::fromStdString(itElFunc->first));
    }

    QString elevationFunctionString = QString::fromStdString(getKeyStringElevationFunction(_interpolationSettings->getChosenElevationFunction()));
    int indexElFunction = elevationFunctionEdit.findData(elevationFunctionString);
    if (indexElFunction != -1)
        elevationFunctionEdit.setCurrentIndex(indexElFunction);

    layoutDetrending->addWidget(&elevationFunctionEdit);


    QVBoxLayout *layoutProxy = new QVBoxLayout;
    QLabel *labelProxy = new QLabel(tr("temperature detrending proxies"));
    layoutProxy->addWidget(labelProxy);

    layoutProxyList = new QVBoxLayout;
    proxyListCheck = new QListWidget;
    layoutProxyList->addWidget(proxyListCheck);
    layoutProxy->addLayout(layoutProxyList);
    redrawProxies();

    QPushButton *editProxy = new QPushButton("Edit proxies...", this);
    layoutProxy->addWidget(editProxy);
    connect(editProxy, &QPushButton::clicked, this, &DialogInterpolation::editProxies);
    layoutDetrending->addLayout(layoutProxy);

    groupDetrending->setLayout(layoutDetrending);
    layoutMainRight->addWidget(groupDetrending);

    // dew point
    QGroupBox *groupRH = new QGroupBox(tr("Relative humidity"));
    QVBoxLayout * layoutRH = new QVBoxLayout();
    useDewPointEdit = new QCheckBox(tr("interpolate relative humidity using dew point"));
    useDewPointEdit->setChecked(_interpolationSettings->getUseDewPoint());
    layoutRH->addWidget(useDewPointEdit);

    // temperature interpolation for relative humidity
    useInterpolTForRH = new QCheckBox(tr("use interpolated temperature (if missing) for relative humidity"));
    useInterpolTForRH->setChecked(_interpolationSettings->getUseInterpolatedTForRH());
    layoutRH->addWidget(useInterpolTForRH);

    groupRH->setLayout(layoutRH);
    layoutMainLeft->addWidget(groupRH);

    //buttons
    QVBoxLayout *layoutMainButtons = new QVBoxLayout();
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layoutMainButtons->addWidget(buttonBox);

    layoutMain->addLayout(layoutMainLeft);
    layoutMain->addLayout(layoutMainRight);
    layoutMain->addLayout(layoutMainButtons);

    //layoutMain->addStretch(1);
    setLayout(layoutMain);

    upscaleFromDemChanged(upscaleFromDemEdit->checkState());
    multipleDetrendingChanged(multipleDetrendingEdit->checkState());
    localDetrendingChanged(localDetrendingEdit->checkState());


    exec();
}

void DialogInterpolation::upscaleFromDemChanged(int active)
{
    gridAggregationMethodEdit.setEnabled(active == Qt::Checked);
}

void DialogInterpolation::multipleDetrendingChanged(int active)
{
    if (active == Qt::Checked) optimalDetrendingEdit->setChecked(Qt::Unchecked);
    optimalDetrendingEdit->setEnabled(active == Qt::Unchecked);
}

void DialogInterpolation::localDetrendingChanged(int active)
{
    if (active == Qt::Checked) topographicDistanceEdit->setChecked(Qt::Unchecked);
    topographicDistanceEdit->setEnabled(active == Qt::Unchecked);
    maxTdMultiplierEdit.setEnabled(active == Qt::Unchecked);
    minPointsLocalDetrendingEdit.setEnabled(active == Qt::Checked);
}

void DialogInterpolation::optimalDetrendingChanged(int active)
{
    if (active == Qt::Checked) localDetrendingEdit->setChecked(Qt::Unchecked);
    localDetrendingEdit->setEnabled(active == Qt::Unchecked);
}

void DialogInterpolation::redrawProxies()
{
    proxyListCheck->clear();
    for (unsigned int i = 0; i < _interpolationSettings->getProxyNr(); i++)
    {
         Crit3DProxy* myProxy = _interpolationSettings->getProxy(i);
         QListWidgetItem *chkProxy = new QListWidgetItem(QString::fromStdString(myProxy->getName()), proxyListCheck);
         chkProxy->setFlags(chkProxy->flags() | Qt::ItemIsUserCheckable);
         if (_interpolationSettings->getSelectedCombination().isProxyActive(i))
            chkProxy->setCheckState(Qt::Checked);
         else
            chkProxy->setCheckState(Qt::Unchecked);
    }
}

void DialogInterpolation::editProxies()
{
    if (_project->meteoPointsDbHandler == nullptr)
    {
        QMessageBox::information(nullptr, "No DB open", "Open DB Points to edit proxy tables and fields");
        return;
    }

    ProxyDialog* myProxyDialog = new ProxyDialog(_project);
    myProxyDialog->close();

    if (myProxyDialog->result() == QDialog::Accepted)
        redrawProxies();
}

void DialogInterpolation::accept()
{
    if (minRegressionR2Edit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing R2", "insert minimum regression R2");
        return;
    }

    if (minPointsLocalDetrendingEdit.text().isEmpty() && localDetrendingEdit->isChecked())
    {
        QMessageBox::information(nullptr, "Missing minimum points number for local detrending", "insert minimum points");
        return;
    }

    if (maxTdMultiplierEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Td multiplier", "insert maximum Td multiplier");
        return;
    }

    if (algorithmEdit.currentIndex() == -1)
    {
        QMessageBox::information(nullptr, "No algorithm selected", "Choose algorithm");
        return;
    }

    for (int i = 0; i < proxyListCheck->count(); i++)
        _interpolationSettings->setActiveSelectedCombination(unsigned(i), proxyListCheck->item(i)->checkState());

    QString algorithmString = algorithmEdit.itemData(algorithmEdit.currentIndex()).toString();
    _interpolationSettings->setInterpolationMethod(interpolationMethodNames.at(algorithmString.toStdString()));

    QString aggrString = gridAggregationMethodEdit.itemData(gridAggregationMethodEdit.currentIndex()).toString();
    _interpolationSettings->setMeteoGridAggrMethod(aggregationMethodToString.at(aggrString.toStdString()));

    _interpolationSettings->setMeteoGridUpscaleFromDem(upscaleFromDemEdit->isChecked());
    _interpolationSettings->setUseTD(topographicDistanceEdit->isChecked());
    _interpolationSettings->setUseLocalDetrending(localDetrendingEdit->isChecked());
    _interpolationSettings->setUseLapseRateCode(lapseRateCodeEdit->isChecked());
    _interpolationSettings->setUseBestDetrending(optimalDetrendingEdit->isChecked());
    _interpolationSettings->setUseMultipleDetrending(multipleDetrendingEdit->isChecked());
    _interpolationSettings->setUseDoNotRetrend(doNotRetrendEdit->isChecked());
    _interpolationSettings->setUseRetrendOnly(retrendOnlyEdit->isChecked());
    _interpolationSettings->setUseThermalInversion(thermalInversionEdit->isChecked());
    _interpolationSettings->setUseDewPoint(useDewPointEdit->isChecked());
    _interpolationSettings->setUseInterpolatedTForRH((useInterpolTForRH->isChecked()));
    _interpolationSettings->setMinRegressionR2(QLocale().toFloat(minRegressionR2Edit.text()));
    _interpolationSettings->setTopoDist_maxKh(maxTdMultiplierEdit.text().toInt());
    _interpolationSettings->setMinPointsLocalDetrending(minPointsLocalDetrendingEdit.text().toInt());

    QString elFunctionString = elevationFunctionEdit.itemData(elevationFunctionEdit.currentIndex()).toString();
    _interpolationSettings->setChosenElevationFunction(fittingFunctionNames.at(elFunctionString.toStdString()));

    _qualityInterpolationSettings->setMinRegressionR2(QLocale().toFloat(minRegressionR2Edit.text()));
    _qualityInterpolationSettings->setTopoDist_maxKh(maxTdMultiplierEdit.text().toInt());
    _qualityInterpolationSettings->setUseLapseRateCode(lapseRateCodeEdit->isChecked());
    _qualityInterpolationSettings->setUseThermalInversion(thermalInversionEdit->isChecked());

    _project->saveInterpolationParameters();

    QDialog::done(QDialog::Accepted);
    return;

}

void ProxyDialog::changedTable()
{
    QSqlDatabase db = _project->meteoPointsDbHandler->getDb();
    _field.clear();
    _field.addItems(getFields(&db, _table.currentText()));
}

void ProxyDialog::changedProxy(bool savePrevious)
{
    if (_proxyCombo.count() == 0) return;

    if (proxyIndex != _proxyCombo.currentIndex() && savePrevious)
    {
        Crit3DProxy *myProxy = &(_proxy.at(unsigned(proxyIndex)));
        saveProxy(myProxy);
    }

    proxyIndex = _proxyCombo.currentIndex();
    Crit3DProxy *myProxy = &(_proxy.at(unsigned(proxyIndex)));

    int index = _table.findText(QString::fromStdString(myProxy->getProxyTable()));
    _table.setCurrentIndex(index);
    index = _field.findText(QString::fromStdString(myProxy->getProxyField()));
    _field.setCurrentIndex(index);
    _proxyGridName.setText(QString::fromStdString(myProxy->getGridName()));
    _forQuality.setChecked(myProxy->getForQualityControl());
}

void ProxyDialog::selectGridFile()
{
    QString qFileName = QFileDialog::getOpenFileName();
    if (qFileName == "") return;
    qFileName = qFileName.left(qFileName.length()-4);

    std::string fileName = qFileName.toStdString();
    std::string error_;
    gis::Crit3DRasterGrid* grid_ = new gis::Crit3DRasterGrid();
    if (gis::readEsriGrid(fileName, grid_, error_))
        _proxyGridName.setText(qFileName);
    else
        QMessageBox::information(nullptr, "Error", "Error opening " + qFileName);

    grid_ = nullptr;
}

void ProxyDialog::listProxies()
{
    _proxyCombo.blockSignals(true);

    _proxyCombo.clear();
    for (unsigned int i = 0; i < _proxy.size(); i++)
         _proxyCombo.addItem(QString::fromStdString(_proxy[i].getName()));

    _proxyCombo.blockSignals(false);
}

void ProxyDialog::addProxy()
{
    bool isValid;
    QString proxyName = QInputDialog::getText(this, tr("New proxy"),
                                         tr("Insert proxy name:"), QLineEdit::Normal,
                                         "", &isValid);
    if (isValid && !proxyName.isEmpty())
    {
        Crit3DProxy myProxy;
        myProxy.setName(proxyName.toStdString());
        _proxy.push_back(myProxy);
        listProxies();
        _proxyCombo.setCurrentIndex(_proxyCombo.count()-1);
    }

    return;
}

void ProxyDialog::deleteProxy()
{
    if (proxyIndex > 0)
        proxyIndex--;
    else
    {
        _table.clear();
        _field.clear();
        _proxyGridName.setText("");
        proxyIndex = NODATA;
    }

    _proxy.erase(_proxy.begin() + _proxyCombo.currentIndex());
    listProxies();
    changedProxy(false);
}

void ProxyDialog::saveProxy(Crit3DProxy* myProxy)
{
    if (_table.currentIndex() != -1)
        myProxy->setProxyTable(_table.currentText().toStdString());

    if (_field.currentIndex() != -1)
        myProxy->setProxyField(_field.currentText().toStdString());

    if (_proxyGridName.text() != "")
        myProxy->setGridName(_proxyGridName.text().toStdString());

    myProxy->setForQualityControl(_forQuality.isChecked());
}

ProxyDialog::ProxyDialog(Project *myProject)
{
    QVBoxLayout *layoutMain = new QVBoxLayout();
    QHBoxLayout *layoutProxyCombo = new QHBoxLayout();
    QVBoxLayout *layoutProxy = new QVBoxLayout();
    QVBoxLayout *layoutPointValues = new QVBoxLayout();
    QVBoxLayout *layoutGrid = new QVBoxLayout();

    this->resize(300, 100);

    _project = myProject;
    _proxy = _project->interpolationSettings.getCurrentProxy();

    setWindowTitle(tr("Interpolation proxy"));

    // proxy list
    QLabel *labelProxyList = new QLabel(tr("proxy list"));
    layoutProxy->addWidget(labelProxyList);
    _proxyCombo.clear();
    listProxies();

    connect(&_proxyCombo, &QComboBox::currentTextChanged, [=](){ this->changedProxy(true); });
    layoutProxyCombo->addWidget(&_proxyCombo);

    QPushButton *_add = new QPushButton("add");
    layoutProxyCombo->addWidget(_add);
    connect(_add, &QPushButton::clicked, [=](){ this->addProxy(); });

    QPushButton *_delete = new QPushButton("delete");
    layoutProxyCombo->addWidget(_delete);
    connect(_delete, &QPushButton::clicked, [=](){ this->deleteProxy(); });

    layoutProxy->addLayout(layoutProxyCombo);
    layoutMain->addLayout(layoutProxy);

    QLabel *labelTableList = new QLabel(tr("table for point values"));
    layoutPointValues->addWidget(labelTableList);
    QList<QString> tables_ = _project->meteoPointsDbHandler->getDb().tables();
    for (int i=0; i < tables_.size(); i++)
        _table.addItem(tables_[i]);

    layoutPointValues->addWidget(&_table);
    connect(&_table, &QComboBox::currentTextChanged, [=](){ this->changedTable(); });

    QLabel *labelFieldList = new QLabel(tr("field for point values"));
    layoutPointValues->addWidget(labelFieldList);
    layoutPointValues->addWidget(&_field);

    layoutMain->addLayout(layoutPointValues);

    // grid
    QLabel *labelGrid = new QLabel(tr("proxy grid"));
    layoutGrid->addWidget(labelGrid);
    QPushButton *_selectGrid = new QPushButton("Select file");
    layoutGrid->addWidget(_selectGrid);
    _proxyGridName.setEnabled(false);
    layoutGrid->addWidget(&_proxyGridName);
    connect(_selectGrid, &QPushButton::clicked, [=](){ this->selectGridFile(); });
    layoutMain->addLayout(layoutGrid);

    // quality control
    _forQuality.setText("use for quality control");
    layoutMain->addWidget(&_forQuality);

    // buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layoutMain->addWidget(buttonBox);

    layoutMain->addStretch(1);
    setLayout(layoutMain);

    proxyIndex = NODATA;

    if (_proxyCombo.count() > 0)
    {
        proxyIndex = 0;
        _proxyCombo.setCurrentIndex(proxyIndex);
        changedProxy(true);
    }

    exec();
}

bool ProxyDialog::checkProxies(QString *error)
{
    QList<QString> fields;
    std::string table_;

    for (unsigned i=0; i < _proxy.size(); i++)
    {
        if (!_project->checkProxy(_proxy[i], error, _project->interpolationSettings.getSelectedCombination().isProxyActive(i)))
            return false;

        table_ = _proxy[i].getProxyTable();
        if (table_ != "")
        {
            QSqlDatabase db = _project->meteoPointsDbHandler->getDb();
            fields = getFields(&db, QString::fromStdString(table_));
            if (fields.filter("id_point").isEmpty())
            {
                *error = "no id_point field found in table " + QString::fromStdString(table_) + " for proxy " + QString::fromStdString(_proxy[i].getName());
                return false;
            }
        }
    }

    return true;
}

void ProxyDialog::saveProxies()
{
    _project->interpolationSettings.initializeProxy();
    _project->qualityInterpolationSettings.initializeProxy();

    std::vector <Crit3DProxy> proxyList;
    std::deque <bool> proxyActive;
    std::vector <int> proxyOrder;

    for (unsigned i=0; i < _proxy.size(); i++)
    {   
        proxyList.push_back(_proxy[i]);
        proxyActive.push_back(true);
        proxyOrder.push_back(i+1);
    }

    _project->addProxyToProject(proxyList, proxyActive, proxyOrder);
}


void ProxyDialog::accept()
{
    if (_proxyCombo.count() > 0)
    {
        Crit3DProxy *myProxy = &(_proxy.at(unsigned(_proxyCombo.currentIndex())));
        saveProxy(myProxy);
    }

    // check proxies
    QString error;
    if (checkProxies(&error))
    {
        saveProxies();
        _project->interpolationSettings.setProxyLoaded(false);

        if (_project->updateProxy())
        {
            QMessageBox::StandardButton reply;
            reply = QMessageBox::question(this, "Save interpolation proxies", "Save changes to settings?",
                                        QMessageBox::Yes|QMessageBox::No);

            if (reply == QMessageBox::Yes)
                _project->saveProxies();

            QDialog::done(QDialog::Accepted);
        }
        else {
            QMessageBox::information(nullptr, "Error updating proxy", error);
        }
    }
    else
        QMessageBox::information(nullptr, "Proxy error", error);

    return;
}
