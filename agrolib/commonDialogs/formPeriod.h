#ifndef FORMPERIOD_H
#define FORMPERIOD_H

    #include <QDialog>
    #include <QDateTime>

    namespace Ui {
        class formPeriod;
    }

    class formPeriod : public QDialog
    {
        Q_OBJECT

    public:
        formPeriod(QDateTime* timeIni, QDateTime* timeFin, QWidget * parent = nullptr);
        ~formPeriod();

        void setMinimumDate(QDate minDate);
        void setMaximumDate(QDate maxDate);

    private slots:
        void on_buttonBox_accepted();

    private:
        Ui::formPeriod *ui;

        QDateTime* dateTimeFirst;
        QDateTime* dateTimeLast;
    };


#endif // FORMPERIOD_H
