
#include "soia.h"

#include "math.h"
#include <algorithm>


double Soia::M(const long& i, const long& j, double& T, Stazione& stazione, const Parametri& parametri)
{
    double L = stazione.Fotoperiodo(i);

    T = std::max(T, b0[j][parametri.varieta]);

    double M = ( a3[j][parametri.varieta] + a1[j][parametri.varieta] * ( L - a0[j][parametri.varieta] )
             + a2[j][parametri.varieta] * (L - a0[j][parametri.varieta] ) * (L - a0[j][parametri.varieta] ) )
             * ( b1[j][parametri.varieta] * ( T - b0[j][parametri.varieta])
             + b2[j][parametri.varieta] * ( T - b0[j][parametri.varieta]) * (T - b0[j][parametri.varieta]));

    if( parametri.varieta == 0) M *= 1.1;
    return M;
}

void Soia::Emergenza(Stazione& stazione, const Parametri& parametri)
{
    long i = m_faseFenologica.size() - 1;
    double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += 1.75 * M( i, 0, Tm, stazione, parametri);	// 1.75 è un fattore correttivo empirico
}

void Soia::Fioritura(Stazione& stazione, const Parametri& parametri)
{
    long i = m_faseFenologica.size() - 1;
    double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += 0.85 * M( i, 1, Tm, stazione, parametri);
}

void Soia::IngrossamentoBaccello(Stazione& stazione, const Parametri& parametri)
{
    long i = m_faseFenologica.size() - 1;
    double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += 1.4 * M( i, 2, Tm, stazione, parametri);
}

void Soia::Maturazione(Stazione& stazione, const Parametri& parametri)
{
    long i = m_faseFenologica.size() - 1;
    double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += 2. * M( i, 3, Tm, stazione, parametri);
}

void Soia::Raccolta(Stazione& stazione, const Parametri& parametri)
{
    long i = m_faseFenologica.size() - 1;
    double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += 2. * M( i, 3, Tm, stazione, parametri);
}


void Soia::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console)
{
    char message[256];

    if( m_faseFenologica.empty() )
    {
        m_faseFenologica.push_back( static_cast<double>(0.) );
    }
    else
    {
        if( m_faseFenologica[m_faseFenologica.size()-1] == parametri.dato_mancante )
        {
            m_faseFenologica[m_faseFenologica.size()-1] = parametri.dato_mancante;
            sprintf(message, "Exception raised: dati fenologici mancanti\n");
            console.Show(message);
            sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
            console.Show(message);
            return;
        }

        m_faseFenologica.push_back( m_faseFenologica[m_faseFenologica.size()-1] );
    }

    if( stazione.Tn(m_faseFenologica.size()-1) == parametri.dato_mancante ||
        stazione.Tx(m_faseFenologica.size()-1) == parametri.dato_mancante )
    {
        sprintf(message, "Exception raised: dati di temperatura mancanti\n");
        console.Show(message);
        sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
        console.Show(message);
        return;
    }

    switch( static_cast<long>( floor(m_faseFenologica[m_faseFenologica.size()-1]) ) )
    {
        case 0:
            if( stazione.getDate(m_faseFenologica.size()-1) == parametri.dataInizio )
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 1.;
                m_semina = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 1:
            Emergenza(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 2.;
                m_emergenza = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 2:
            Fioritura(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 3.;
                m_inizioFioritura = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 3:
            Fioritura(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 4.)
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 4.;
                m_fineFioritura = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 4:
            IngrossamentoBaccello(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 5.)
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 5.;
                m_ingrossamentoBaccello = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 5:
            Maturazione(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 6. )
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 6.;
                m_maturazione = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        case 6:
            Raccolta(stazione, parametri);

            if( m_faseFenologica[m_faseFenologica.size()-1] >= 7. )
            {
                m_faseFenologica[m_faseFenologica.size()-1] = 7.;
                m_raccolta = stazione.getDate(m_faseFenologica.size()-1);
            }

            break;

        default:
            // nessuna azione;
            break;
    }
}

double Soia::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
{
    double bbch = m_faseFenologica[giorno];

    if( parametri.scalaBBCH )
    {
        switch( static_cast<long>( floor(m_faseFenologica[giorno]) ) )
        {
            case 1:
                bbch = 9. * ( m_faseFenologica[giorno] - 1. );
                break;

            case 2:
                bbch = 9. + ( 60. - 9. ) * ( m_faseFenologica[giorno] - 2. );
                break;

            case 3:
                bbch = 60. + ( 69. - 60. ) * ( m_faseFenologica[giorno] - 3. );
                break;

            case 4:
                bbch = 69. + ( 75. - 69. ) * ( m_faseFenologica[giorno] - 4. );
                break;

            case 5:
                bbch = 75. + ( 89. - 75. ) * ( m_faseFenologica[giorno] - 5. );
                break;

            case 6:
                bbch = 89. + ( 99. - 89. ) * ( m_faseFenologica[giorno] - 6. );
                break;

            case 7:
                bbch = 99.;
                break;

            default:
                // nessuna azione;
                bbch = parametri.dato_mancante;
        }
    }

    return bbch;
}
