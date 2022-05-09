#ifndef DIALOGCHANGEAXIS_H
#define DIALOGCHANGEAXIS_H

    #include <QtWidgets>

    class DialogChangeAxis : public QDialog
    {
        Q_OBJECT

    private:
        bool isLeftAxis;
        QLineEdit minVal;
        QLineEdit maxVal;

    public:
        DialogChangeAxis(bool isLeftAxis);
        ~DialogChangeAxis() override;
        void done(bool res);
        float getMinVal() const;
        float getMaxVal() const;
    };

#endif // DIALOGCHANGEAXIS_H
