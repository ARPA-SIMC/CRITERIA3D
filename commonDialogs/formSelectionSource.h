#ifndef FORMSELECTIONSOURCE_H
#define FORMSELECTIONSOURCE_H

    #include <QtWidgets>

    class FormSelectionSource : public QDialog
    {
        Q_OBJECT

    public:
        FormSelectionSource();

        int getSourceSelectionId();

    private:
        QRadioButton* pointButton;
        QRadioButton* gridButton;

        void done(int res);
    };

#endif // FORMSELECTIONSOURCE_H
