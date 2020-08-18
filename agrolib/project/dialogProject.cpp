#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#include <QDialogButtonBox>

#include "dialogProject.h"
#include "project.h"


DialogProject::DialogProject(Project *myProject)
{
    project_ = myProject;

    QVBoxLayout *layoutMain = new QVBoxLayout();

    this->resize(300, 100);

    setWindowTitle(tr("New project"));

    // name
    QLabel *labelProjectFileName = new QLabel(tr("Name"));
    layoutMain->addWidget(labelProjectFileName);
    lineEditProjectName = new QLineEdit(tr("myNewProject"));
    layoutMain->addWidget(lineEditProjectName);

    // path
    QLabel *labelProjectPath = new QLabel(tr("Path"));
    layoutMain->addWidget(labelProjectPath);
    lineEditProjectPath = new QLineEdit(project_->getDefaultPath() + PATH_PROJECT);
    layoutMain->addWidget(lineEditProjectPath);
    QPushButton* buttonSelectPath = new QPushButton("Select project path");
    layoutMain->addWidget(buttonSelectPath);
    lineEditProjectPath->setEnabled(false);
    connect(buttonSelectPath, &QPushButton::clicked, [=](){ this->getPath(); });

    // description
    QLabel *labelProjectName = new QLabel(tr("Description"));
    layoutMain->addWidget(labelProjectName);
    lineEditProjectDescription = new QLineEdit(tr(""));
    layoutMain->addWidget(lineEditProjectDescription);

    // buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    layoutMain->addWidget(buttonBox);

    layoutMain->addStretch(1);
    setLayout(layoutMain);
}


void DialogProject::getPath()
{
    lineEditProjectPath->setText(QFileDialog::getExistingDirectory(this, "", project_->getDefaultPath() + PATH_PROJECT));
}

void DialogProject::accept()
{
    if (lineEditProjectName->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing name", "Choose project name");
        return;
    }

    if (lineEditProjectPath->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing path", "Choose project path");
        return;
    }

    project_->createProject(lineEditProjectPath->text(), lineEditProjectName->text(),
                            lineEditProjectDescription->text());

    QDialog::done(QDialog::Accepted);
}
