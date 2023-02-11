#include "dialogNewProject.h"

DialogNewProject::DialogNewProject()
{
    setWindowTitle("New Project");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *layoutProject = new QHBoxLayout();
    QGridLayout *layoutDb = new QGridLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout();

    QLabel *projectNameLabel = new QLabel(tr("Enter project name (without spaces): "));
    projectName = new QLineEdit();

    layoutProject->addWidget(projectNameLabel);
    layoutProject->addWidget(projectName);

    layoutDb->addWidget(createSoilGroup(), 0, 0);
    layoutDb->addWidget(createMeteoGroup(), 0, 1);
    layoutDb->addWidget(createCropGroup(), 1, 0);


    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);


    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);


    mainLayout->addLayout(layoutProject);
    mainLayout->addLayout(layoutDb);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);

    show();
    exec();

}

QGroupBox *DialogNewProject::createSoilGroup()
{

    QGroupBox *groupBox = new QGroupBox(tr("Select SOIL db"));

    newSoil = new QRadioButton(tr("&New Db"));
    defaultSoil = new QRadioButton(tr("&Default Db (ER soil)"));
    chooseSoil = new QRadioButton(tr("&Choose Db"));

    newSoil->setChecked(true);


    QVBoxLayout *vbox = new QVBoxLayout;
    vbox->addWidget(newSoil);
    vbox->addWidget(defaultSoil);
    vbox->addWidget(chooseSoil);
    dbSoilName = new QLineEdit();
    vbox->addWidget(dbSoilName);
    dbSoilName->setVisible(false);
    vbox->addStretch(1);
    groupBox->setLayout(vbox);
    connect(newSoil, &QRadioButton::clicked, [=](){ this->hideSoilName(); });
    connect(defaultSoil, &QRadioButton::clicked, [=](){ this->hideSoilName(); });
    connect(chooseSoil, &QRadioButton::clicked, [=](){ this->chooseSoilDb(); });

    return groupBox;
}

QGroupBox *DialogNewProject::createMeteoGroup()
{

    QGroupBox *groupBox = new QGroupBox(tr("Select METEO db"));

    newMeteo = new QRadioButton(tr("&New Db"));
    defaultMeteo = new QRadioButton(tr("&Default Db (test data)"));
    chooseMeteo = new QRadioButton(tr("&Choose Db"));

    newMeteo->setChecked(true);


    QVBoxLayout *vbox = new QVBoxLayout;
    vbox->addWidget(newMeteo);
    vbox->addWidget(defaultMeteo);
    vbox->addWidget(chooseMeteo);
    dbMeteoName = new QLineEdit();
    vbox->addWidget(dbMeteoName);
    dbMeteoName->setVisible(false);
    vbox->addStretch(1);
    groupBox->setLayout(vbox);
    connect(newMeteo, &QRadioButton::clicked, [=](){ this->hideMeteoName(); });
    connect(defaultMeteo, &QRadioButton::clicked, [=](){ this->hideMeteoName(); });
    connect(chooseMeteo, &QRadioButton::clicked, [=](){ this->chooseMeteoDb(); });

    return groupBox;
}

QGroupBox *DialogNewProject::createCropGroup()
{

    QGroupBox *groupBox = new QGroupBox(tr("Select CROP parameters db"));

    defaultCrop = new QRadioButton(tr("&Default Db"));
    chooseCrop = new QRadioButton(tr("&Choose Db"));

    defaultCrop->setChecked(true);


    QVBoxLayout *vbox = new QVBoxLayout;
    vbox->addWidget(defaultCrop);
    vbox->addWidget(chooseCrop);
    dbCropName = new QLineEdit();
    vbox->addWidget(dbCropName);
    dbCropName->setVisible(false);
    vbox->addStretch(1);
    groupBox->setLayout(vbox);

    connect(defaultCrop, &QRadioButton::clicked, [=](){ this->hideCropName(); });
    connect(chooseCrop, &QRadioButton::clicked, [=](){ this->chooseCropDb(); });

    return groupBox;
}

void DialogNewProject::done(int res)
{
    if(res)  // ok was pressed
    {
        if (projectName->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing project name", "Insert project name");
            return;
        }
        QDialog::done(QDialog::Accepted);
        return;

    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}

QString DialogNewProject::getProjectName() const
{
    return projectName->text();
}

QString DialogNewProject::getDbSoilCompletePath() const
{
    return dbSoilCompletePath;
}

QString DialogNewProject::getDbMeteoCompletePath() const
{
    return dbMeteoCompletePath;
}

QString DialogNewProject::getDbCropCompletePath() const
{
    return dbCropCompletePath;
}

void DialogNewProject::chooseSoilDb()
{
    dbSoilCompletePath = QFileDialog::getOpenFileName(this, tr("Open soil db"), "", tr("SQLite files (*.db)"));
    if (dbSoilCompletePath == "")
    {
        newSoil->setChecked(true);
        return;
    }
    chooseSoil->setVisible(false);
    dbSoilName->setVisible(true);
    dbSoilName->setText(QFileInfo(dbSoilCompletePath).baseName());
}

void DialogNewProject::chooseMeteoDb()
{
    dbMeteoCompletePath = QFileDialog::getOpenFileName(this, tr("Open meteo db"), "", tr("SQLite files (*.db)"));
    if (dbMeteoCompletePath == "")
    {
        newMeteo->setChecked(true);
        return;
    }
    chooseMeteo->setVisible(false);
    dbMeteoName->setVisible(true);
    dbMeteoName->setText(QFileInfo(dbMeteoCompletePath).baseName());
}

void DialogNewProject::chooseCropDb()
{
    dbCropCompletePath = QFileDialog::getOpenFileName(this, tr("Open crop db"), "", tr("SQLite files (*.db)"));
    if (dbCropCompletePath == "")
    {
        defaultCrop->setChecked(true);
        return;
    }
    chooseCrop->setVisible(false);
    dbCropName->setVisible(true);
    dbCropName->setText(QFileInfo(dbCropCompletePath).baseName());
}

void DialogNewProject::hideSoilName()
{
    chooseSoil->setVisible(true);
    dbSoilName->setVisible(false);
    dbSoilName->setText("");
}

void DialogNewProject::hideMeteoName()
{
    chooseMeteo->setVisible(true);
    dbMeteoName->setVisible(false);
    dbMeteoName->setText("");
}

void DialogNewProject::hideCropName()
{
    chooseCrop->setVisible(true);
    dbCropName->setVisible(false);
    dbCropName->setText("");
}


int DialogNewProject::getSoilDbOption()
{
    if (newSoil->isChecked())
    {
        return NEW_DB;
    }
    else if(defaultSoil->isChecked())
    {
        return DEFAULT_DB;
    }
    else if(chooseSoil->isChecked())
    {
        return CHOOSE_DB;
    }
    else
    {
        return NEW_DB;
    }
}


int DialogNewProject::getMeteoDbOption()
{
    if (newMeteo->isChecked())
    {
        return NEW_DB;
    }
    else if(defaultMeteo->isChecked())
    {
        return DEFAULT_DB;
    }
    else if(chooseMeteo->isChecked())
    {
        return CHOOSE_DB;
    }
    else
    {
        return NEW_DB;
    }
}


int DialogNewProject::getCropDbOption()
{
    if(defaultCrop->isChecked())
    {
        return DEFAULT_DB;
    }
    else if(chooseCrop->isChecked())
    {
        return CHOOSE_DB;
    }
    else
    {
        return DEFAULT_DB;
    }
}
