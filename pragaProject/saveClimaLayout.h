#ifndef SAVECLIMALAYOUT_H
#define SAVECLIMALAYOUT_H

#include <QString>
#include <QGridLayout>
#include <QtWidgets>

bool compareClimateElab(const QString &el1, const QString &el2);

class SaveClimaLayout : public QDialog
{

    Q_OBJECT

    public:
        SaveClimaLayout();

        QString getFirstYear() const;
        void setFirstYear(const QString &value);
        QString getLastYear() const;
        void setLastYear(const QString &value);
        QString getVariable() const;
        void setVariable(const QString &value);
        QString getPeriod() const;
        void setPeriod(const QString &value);
        QString getGenericPeriodEnd() const;
        void setGenericPeriodEnd(const QString &value);
        QString getGenericNYear() const;
        void setGenericNYear(const QString &value);
        QString getSecondElab() const;
        void setSecondElab(const QString &value);
        QString getElab2Param() const;
        void setElab2Param(const QString &value);
        QString getElab() const;
        void setElab(const QString &value);
        QString getElab1Param() const;
        void setElab1Param(const QString &value);
        QString getGenericPeriodStartDay() const;
        void setGenericPeriodStartDay(const QString &value);
        QString getGenericPeriodStartMonth() const;
        void setGenericPeriodStartMonth(const QString &value);
        QString getGenericPeriodEndDay() const;
        void setGenericPeriodEndDay(const QString &value);
        QString getGenericPeriodEndMonth() const;
        void setGenericPeriodEndMonth(const QString &value);

        void addElab();
        void deleteRaw();
        void deleteAll();

        void saveElabList();
        void loadElabList();

        QList<QString> getList() const;
        void setList(const QList<QString> &value);

        QString getElab1ParamFromdB() const;
        void setElab1ParamFromdB(const QString &value);

private:

        QVBoxLayout mainLayout;
        QVBoxLayout listLayout;
        QHBoxLayout saveButtonLayout;

        QListWidget listView;
        QList<QString> list;

        QPushButton saveList;
        QPushButton loadList;

        QString firstYear;
        QString lastYear;
        QString variable;
        QString period;

        QString genericPeriodStartDay;
        QString genericPeriodStartMonth;
        QString genericPeriodEndDay;
        QString genericPeriodEndMonth;
        QString genericNYear;

        QString secondElab;
        QString elab2Param;

        QString elab;
        QString elab1Param;
        QString elab1ParamFromdB;


};


#endif // SAVECLIMALAYOUT_H
