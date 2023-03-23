#ifndef FORMSELECTIONSOURCE_H
#define FORMSELECTIONSOURCE_H

    #include <QtWidgets>

    class FormSelectionSource : public QDialog
    {
        Q_OBJECT

    public:
        FormSelectionSource();
        void done(int res);
        int getSourceSelectionId();

    private:
        QRadioButton* pointButton;
        QRadioButton* gridButton;
    };

#endif // FORMSELECTIONSOURCE_H
