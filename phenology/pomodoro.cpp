
#include "pomodoro.h"

#include "math.h"
#include <algorithm>

double Pomodoro::Dday(Stazione& stazione, const long& i)
{
	double L = stazione.Fotoperiodo(i);
	double Td = Tday(stazione.Tn(i), stazione.Tx(i));
    return std::max( static_cast<double>(0.),
				  ( -6.0304 + 0.5408 * Td - 0.0104 * Td * Td ) * L / 24. );
}

double Pomodoro::Dday(Stazione& stazione, const long& i, const double& Ts) 
{
	double L = stazione.Fotoperiodo(i);
    return std::max( static_cast<double>(0.),
		          ( Tday(stazione.Tn(i), stazione.Tx(i)) - Ts ) / ( 26. - Ts) * L / 24. );
}

double Pomodoro::Dnyt(Stazione& stazione, const long& i, const double& Ts)
{
	double L = stazione.Fotoperiodo(i);
    return std::max( static_cast<double>(0.),
		          ( Tnyt(stazione.Tn(i), stazione.Tx(i)) - Ts ) / ( 26. - Ts) * ( 24. - L ) / 24. );
}


void Pomodoro::Emergenza(Stazione& stazione)
{    
	double Ts = 8.;
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += ( Dday(stazione, i, Ts) + Dnyt(stazione, i, Ts) ) / 4.;
}


void Pomodoro::PrimoFiore(Stazione& stazione)
{    
	double Ts = 10.;
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += ( Dday(stazione, i, Ts) + Dnyt(stazione, i, Ts) ) / 25.;
}

void Pomodoro::Invaiatura(Stazione& stazione) 
{
	double Ts = 10.;
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += ( Dday(stazione, i) + Dnyt(stazione, i, Ts) ) / 35.;

    if(Tnyt( stazione.Tn(i), stazione.Tx(i) ) > 10. && Tnyt( stazione.Tn(i), stazione.Tx(i) ) < 15. )
		m_faseFenologica[i] += .25 / 35.;
}


void Pomodoro::Maturazione(Stazione& stazione) 
{
	double	Ts = 10.;
	double	limitePHD = .5 * ( m_minPHD + m_maxPHD );

	// si tiene conto dello stato idrico del suolo
	long i = m_faseFenologica.size() - 1;
	if( stazione.Pr(i) < 1. )
	{
		m_giorniSiccitosi++;
		m_accumuloPrec += 0.;
		limitePHD = m_giorniSiccitosi < 20  ? limitePHD : m_minPHD;
	}
	else
	{
		m_giorniSiccitosi = 0;
		m_accumuloPrec += stazione.Pr(i);
		limitePHD = m_accumuloPrec < 100. ? limitePHD : m_maxPHD;
    }

    m_faseFenologica[i] += ( Dday(stazione, i) + Dnyt(stazione, i, Ts) ) / limitePHD;

    if(Tnyt( stazione.Tn(i), stazione.Tx(i) ) > 10. && Tnyt( stazione.Tn(i), stazione.Tx(i) ) < 15. )
		m_faseFenologica[i] += .25 / limitePHD;
}

void Pomodoro::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
			Emergenza(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_emergenza = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 2:
			PrimoFiore(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_primoFiore = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Invaiatura(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_invaiatura = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 4:
			Maturazione(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 5. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 5.;
				m_maturazione = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double Pomodoro::FaseFenologica(const unsigned long& giorno, const Parametri& parametri) 
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
				bbch = 9. + ( 61. - 9. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 61. + ( 84. - 61. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 84. + ( 99. - 84. ) * ( m_faseFenologica[giorno] - 4. );
				break;

			case 5:
				bbch = 99.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}


//
// pomodoro da trapianto
//

void PomodoroTrapianto::PrimoFiore(Stazione& stazione) 
{
	double Ts = 10.;
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += ( Dday(stazione, i, Ts) + Dnyt(stazione, i, Ts) ) / 12.;

    if(Tnyt( stazione.Tn(i), stazione.Tx(i) ) > 10. && Tnyt( stazione.Tn(i), stazione.Tx(i) ) < 15. )
		m_faseFenologica[i] += .25 / 12.;
}
  
void PomodoroTrapianto::Invaiatura(Stazione& stazione) 
{
	double Ts = 10.;
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += ( Dday(stazione, i) + Dnyt(stazione, i, Ts) ) / 40.;

    if(Tnyt( stazione.Tn(i), stazione.Tx(i) ) > 10. && Tnyt( stazione.Tn(i), stazione.Tx(i) ) < 15. )
		m_faseFenologica[i] += .25 / 40.;
}

void PomodoroTrapianto::Maturazione(Stazione& stazione) 
{
	double	Ts = 10.;
	double	limitePHD = .5 * ( m_minPHD + m_maxPHD );

	// si tiene conto dello stato idrico del suolo
	long i = m_faseFenologica.size() - 1;
	if( stazione.Pr(i) < 1. )
	{
		m_giorniSiccitosi++;
		m_accumuloPrec += 0.;
		limitePHD = m_giorniSiccitosi < 20  ? limitePHD : m_minPHD;
	}
	else
	{
		m_giorniSiccitosi = 0;
		m_accumuloPrec += stazione.Pr(i);
		limitePHD = m_accumuloPrec < 100. ? limitePHD : m_maxPHD;
    }


    m_faseFenologica[i] += ( Dday(stazione, i) + Dnyt(stazione, i, Ts) ) / limitePHD;

    if(Tnyt( stazione.Tn(i), stazione.Tx(i) ) > 10. && Tnyt( stazione.Tn(i), stazione.Tx(i) ) < 15. )
		m_faseFenologica[i] += .25 / limitePHD;
}


void PomodoroTrapianto::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_trapianto = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 1:
			if( m_faseFenologica[m_faseFenologica.size()-1] >= 1. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_trapianto = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 2:
			PrimoFiore(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_primoFiore = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Invaiatura(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_invaiatura = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 4:
			Maturazione(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 5. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 5.;
				m_maturazione = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double PomodoroTrapianto::FaseFenologica(const unsigned long& giorno, const Parametri& parametri) 
{ 
	double bbch = m_faseFenologica[giorno];

	if( parametri.scalaBBCH )
	{
		switch( static_cast<long>( floor(m_faseFenologica[giorno]) ) )
		{
			case 1:
				bbch = 13. * ( m_faseFenologica[giorno] - 1. );
				break;

			case 2:
				bbch = 13. + ( 61. - 13. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 61. + ( 84. - 61. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 84. + ( 99. - 84. ) * ( m_faseFenologica[giorno] - 4. );
				break;

			case 5:
				bbch = 99.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}

