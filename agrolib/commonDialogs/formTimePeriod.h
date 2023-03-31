#ifndef FORMTIMEPERIOD_H
#define FORMTIMEPERIOD_H

    #include <QDialog>
    class QDateTime;

    namespace Ui {
        class frmTimePeriod;
    }

    class FormTimePeriod : public QDialog
    {
        Q_OBJECT

    public:
        FormTimePeriod(QDateTime* timeIni, QDateTime* timeFin, QWidget * parent = nullptr);

        void setMinimumDate(QDate myDate);
        void setMaximumDate(QDate myDate);

    private slots:
        void on_buttonBox_accepted();

    private:
        Ui::frmTimePeriod *ui;

        QDateTime* dateTimeFirst;
        QDateTime* dateTimeLast;
    };


#endif // FORMTIMEPERIOD_H
