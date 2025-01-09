#ifndef SOILPHYSICS_H
#define SOILPHYSICS_H

    struct Tsoil;

    double computeWaterConductivity(double Se, Tsoil *mySoil);
    double computeSefromPsi_unsat(double psi, Tsoil *mySoil);
    double theta_from_Se(unsigned long index);
    double theta_from_Se (double Se, unsigned long index);
    double theta_from_sign_Psi (double myPsi, unsigned long index);
    double Se_from_theta (unsigned long index, double myTheta);
    double psi_from_Se(unsigned long index);
    double computeSe(unsigned long index);
    double dTheta_dH(unsigned long index);
    double dThetav_dH(unsigned long index, double temperature, double dTheta_dH);
    double computeK(unsigned long index);
    double compute_K_Mualem(double Ksat, double Se, double VG_Sc, double VG_m, double Mualem_L);
    double getThetaMean(long i);
    double getTheta(long i, double H);
    double getHMean(long i);
    double getPsiMean(long i);
    double estimateBulkDensity(long i);
    double getTMean(long i);

#endif  // SOILPHYSICS_H
