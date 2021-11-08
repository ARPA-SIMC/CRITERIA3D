#ifndef FORMPERIOD_H
#define FORMPERIOD_H

    #include <QDialog>
    class QDateTime;

    namespace Ui {
        class FormPeriod;
    }

    class FormPeriod : public QDialog
    {
        Q_OBJECT

    public:
        FormPeriod(QDateTime* timeIni, QDateTime* timeFin, QWidget * parent = nullptr);
        ~FormPeriod() override;

        void setMinimumDate(QDate myDate);
        void setMaximumDate(QDate myDate);

    private slots:
        void on_buttonBox_accepted();

    private:
        Ui::FormPeriod *ui;

        QDateTime* dateTimeFirst;
        QDateTime* dateTimeLast;
    };


#endif // FORMPERIOD_H
