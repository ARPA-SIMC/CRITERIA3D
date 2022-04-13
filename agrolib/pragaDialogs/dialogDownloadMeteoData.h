#ifndef DIALOGDOWNLOADMETEODATA_H
#define DIALOGDOWNLOADMETEODATA_H

#include <QtWidgets>

    class PragaProject;

    class DialogDownloadMeteoData : public QDialog
    {

        Q_OBJECT

    private:
        QListWidget dailyVar;
        QListWidget hourlyVar;
        QDateEdit firstYearEdit;
        QDateEdit lastYearEdit;
        QListWidgetItem daily_item1;
        QListWidgetItem daily_item2;
        QListWidgetItem daily_item3;
        QListWidgetItem daily_item4;
        QListWidgetItem daily_item5;
        QListWidgetItem daily_item6;
        QListWidgetItem daily_item7;
        QListWidgetItem daily_item8;
        QListWidgetItem daily_item9;
        QListWidgetItem daily_item10;
        QListWidgetItem daily_item11;

        QListWidgetItem hourly_item1;
        QListWidgetItem hourly_item2;
        QListWidgetItem hourly_item3;
        QListWidgetItem hourly_item4;
        QListWidgetItem hourly_item5;
        QListWidgetItem hourly_item6;

        QCheckBox allDaily;
        QCheckBox allHourly;

        QList<QString> varD;
        QList<QString> varH;
        bool prec0024;

    public:
        DialogDownloadMeteoData();
        void done(bool res);
        void allDailyVarClicked(int toggled);
        void allHourlyVarClicked(int state);
        void dailyItemClicked(QListWidgetItem * item);
        void hourlyItemClicked(QListWidgetItem * item);
        QList<QString> getVarD() const;
        QList<QString> getVarH() const;
        QDate getFirstDate();
        QDate getLastDate();
        bool getPrec0024() const;
    };

#endif // DIALOGDOWNLOADMETEODATA_H
