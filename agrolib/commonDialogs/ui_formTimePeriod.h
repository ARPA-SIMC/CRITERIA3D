/********************************************************************************
** Form generated from reading UI file 'formTimePeriod.ui'
**
** Created by: Qt User Interface Compiler version 6.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FORMTIMEPERIOD_H
#define UI_FORMTIMEPERIOD_H

#include <QtCore/QDate>
#include <QtCore/QVariant>
#include <QtWidgets/QAbstractButton>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDateTimeEdit>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QLabel>

QT_BEGIN_NAMESPACE

class Ui_frmTimePeriod
{
public:
    QDialogButtonBox *buttonBox;
    QDateTimeEdit *dateTimeEditFirst;
    QDateTimeEdit *dateTimeEditLast;
    QLabel *label;
    QLabel *label_2;

    void setupUi(QDialog *frmTimePeriod)
    {
        if (frmTimePeriod->objectName().isEmpty())
            frmTimePeriod->setObjectName("frmTimePeriod");
        frmTimePeriod->resize(282, 180);
        buttonBox = new QDialogButtonBox(frmTimePeriod);
        buttonBox->setObjectName("buttonBox");
        buttonBox->setGeometry(QRect(60, 130, 171, 41));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        dateTimeEditFirst = new QDateTimeEdit(frmTimePeriod);
        dateTimeEditFirst->setObjectName("dateTimeEditFirst");
        dateTimeEditFirst->setGeometry(QRect(60, 40, 171, 22));
        dateTimeEditFirst->setDateTime(QDateTime(QDate(2013, 12, 31), QTime(22, 0, 0)));
        dateTimeEditFirst->setTime(QTime(22, 0, 0));
        dateTimeEditFirst->setMinimumDateTime(QDateTime(QDate(1900, 12, 31), QTime(22, 0, 0)));
        dateTimeEditFirst->setCalendarPopup(true);
        dateTimeEditFirst->setTimeSpec(Qt::UTC);
        dateTimeEditLast = new QDateTimeEdit(frmTimePeriod);
        dateTimeEditLast->setObjectName("dateTimeEditLast");
        dateTimeEditLast->setGeometry(QRect(60, 90, 171, 22));
        dateTimeEditLast->setDateTime(QDateTime(QDate(2014, 1, 1), QTime(22, 0, 0)));
        dateTimeEditLast->setDate(QDate(2014, 1, 1));
        dateTimeEditLast->setCalendarPopup(true);
        dateTimeEditLast->setTimeSpec(Qt::UTC);
        label = new QLabel(frmTimePeriod);
        label->setObjectName("label");
        label->setGeometry(QRect(60, 20, 71, 16));
        label_2 = new QLabel(frmTimePeriod);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(60, 70, 81, 16));

        retranslateUi(frmTimePeriod);
        QObject::connect(buttonBox, &QDialogButtonBox::accepted, frmTimePeriod, qOverload<>(&QDialog::accept));
        QObject::connect(buttonBox, &QDialogButtonBox::rejected, frmTimePeriod, qOverload<>(&QDialog::reject));

        QMetaObject::connectSlotsByName(frmTimePeriod);
    } // setupUi

    void retranslateUi(QDialog *frmTimePeriod)
    {
        frmTimePeriod->setWindowTitle(QCoreApplication::translate("frmTimePeriod", "Select period", nullptr));
        dateTimeEditFirst->setDisplayFormat(QCoreApplication::translate("frmTimePeriod", "yyyy-MM-dd HH.mm", nullptr));
        dateTimeEditLast->setDisplayFormat(QCoreApplication::translate("frmTimePeriod", "yyyy-MM-dd HH.mm", nullptr));
        label->setText(QCoreApplication::translate("frmTimePeriod", "First date:", nullptr));
        label_2->setText(QCoreApplication::translate("frmTimePeriod", "Last date:", nullptr));
    } // retranslateUi

};

namespace Ui {
    class frmTimePeriod: public Ui_frmTimePeriod {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FORMTIMEPERIOD_H
