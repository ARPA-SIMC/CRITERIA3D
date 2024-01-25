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

        QString cellListFileName;

        QCheckBox allDaily;
        QCheckBox allHourly;

        std::vector<QListWidgetItem> dailyItems;
        std::vector<QListWidgetItem> hourlyItems;

        QList<QString> dailyVariableList;
        QList<QString> hourlyVariableList;

        void allDailyVarClicked(int toggled);
        void allHourlyVarClicked(int state);

        void dailyItemClicked(QListWidgetItem * item);
        void hourlyItemClicked(QListWidgetItem * item);

        void on_actionLoadCellList();

        void done(bool result);

    public:
        DialogExportDataGrid();

        QList<QString> getDailyVariableList() const
            { return dailyVariableList; }

        QList<QString> getHourlyVariableList() const
            { return hourlyVariableList; }

        QDate getFirstDate()
            { return firstDateEdit.date(); }

        QDate getLastDate()
            { return lastDateEdit.date(); }

        QString getCellListFileName()
            { return cellListFileName; }
    };


#endif // DIALOGEXPORTDATAGRID_H
