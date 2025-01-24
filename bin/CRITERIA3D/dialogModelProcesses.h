#ifndef DIALOGMODELPROCESSES_H
#define DIALOGMODELPROCESSES_H

    #include <QDialog>
    #include <QCheckBox>

    class DialogModelProcesses : public QDialog
    {
    private:

    public:
        QCheckBox *hydrallProcess;
        QCheckBox *snowProcess;
        QCheckBox *cropProcess;
        QCheckBox *waterFluxesProcess;

        DialogModelProcesses();
    };


#endif // DIALOGMODELPROCESSES_H
