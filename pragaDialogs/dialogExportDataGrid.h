#ifndef DIALOGEXPORTDATAGRID_H
#define DIALOGEXPORTDATAGRID_H

#include <QtWidgets>

    class DialogExportDataGrid : public QDialog
    {
        Q_OBJECT

    private:
        QListWidget dailyVar;
        QListWidget hourlyVar;

        QDateEdit firstDateEdit;
        QDateEdit lastDateEdit;

        std::vector<QListWidgetItem> dailyItems;
        std::vector<QListWidgetItem> hourlyItems;

        QCheckBox allDaily;
        QCheckBox allHourly;

        QList<QString> dailyVariableList;
        QList<QString> hourlyVariableList;

    public:
        DialogExportDataGrid();

        void allDailyVarClicked(int toggled);
        void allHourlyVarClicked(int state);

        void dailyItemClicked(QListWidgetItem * item);
        void hourlyItemClicked(QListWidgetItem * item);

        void done(bool result);

        QList<QString> getDailyVariableList() const
        { return dailyVariableList; }

        QList<QString> getHourlyVariableList() const
        { return hourlyVariableList; }

        QDate getFirstDate()
        { return firstDateEdit.date(); }

        QDate getLastDate()
        { return lastDateEdit.date(); }
    };


#endif // DIALOGEXPORTDATAGRID_H
