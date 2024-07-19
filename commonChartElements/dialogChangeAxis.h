#ifndef DIALOGCHANGEAXIS_H
#define DIALOGCHANGEAXIS_H

    #include <QtWidgets>

    class DialogChangeAxis : public QDialog
    {
        Q_OBJECT

    private:
        QDateEdit minDate;
        QLineEdit minVal;

        QDateEdit maxDate;
        QLineEdit maxVal;

    public:
        DialogChangeAxis(int nrAxis, bool isDate);
        ~DialogChangeAxis() override;
        void done(bool res);

        float getMinVal() const;
        float getMaxVal() const;

        QDate getMinDate() const;
        QDate getMaxDate() const;
    };

#endif // DIALOGCHANGEAXIS_H
