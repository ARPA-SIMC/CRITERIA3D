#include "cereali.h"
#include "float.h"
#include "math.h"

// calcolo delle date di semina ed emergenza fittizie
bool Cereali::CalcoloDateFittizie(const Stazione& stazione, const Parametri& parametri, Console& console)
{
	char message[256];

	long i;
	long k = 0;

	double vernalizzazione = DBL_MAX;

	for( i = 2; i < stazione.NumeroGiorni() - 2; i++ )
	{
        if( stazione.getDate(i) <= m_semina )
			continue;

		double Tm = 0.;

		for( long j = -2; j < 3; j++ )
		{
			if( stazione.Tn(i + j) == parametri.dato_mancante || stazione.Tx(i + j) == parametri.dato_mancante )
			{
				sprintf(message, "Exception raised: dati di temperatura mancanti\n");
				console.Show(message);
				sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
				console.Show(message);
				return false;
			}

			Tm += ( stazione.Tn(i + j) + stazione.Tx(i + j) ) / 10.;
		}

		if ( Tm < vernalizzazione ) 
		{
			k = i;
			vernalizzazione = Tm;
            m_seminaFittizia = stazione.getDate(i);
		}
	}

	if( vernalizzazione >= parametri.sogliaVernalizzazione )
	{
		sprintf(message, "Exception raised: semina fittizia non calcolata\n");
		console.Show(message);
		sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
		console.Show(message);
		return false;
	}

	double sommaTermica = 0.;

	for( i = k; i < stazione.NumeroGiorni(); i++ )
	{
		double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

        sommaTermica += std::max( static_cast<double>(0.), TassoSviluppoEmergenza(Tm) );
			
		if( sommaTermica >= 1. )
		{
			k = i;
			break;
		}
	}

	if( k >= stazione.NumeroGiorni() )
	{
		sprintf(message, "Exception raised: emergenza fittizia non calcolata\n");
		console.Show(message);
		sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
		console.Show(message);
		return false;
	}

    m_emergenzaFittizia = stazione.getDate(k);
	return true;
}

// calcolo del numero di foglie totali
bool Cereali::CalcoloNumeroFoglie(Stazione& stazione, const Parametri& parametri, Console& console) 
{	
	char message[256];

	long indexEmergenza = 0;

	double P = 4.;

	for( long i = 0; i < stazione.NumeroGiorni(); i++ )
	{
		if( stazione.Tn(i) == parametri.dato_mancante || stazione.Tx(i) == parametri.dato_mancante )
		{
			sprintf(message, "Exception raised: dati di temperatura mancanti\n");
			console.Show(message);
			sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
			console.Show(message);
			return false;
		}			

        if( stazione.getDate(i) <= m_semina )
			continue;

        if( stazione.getDate(i) > m_emergenzaFittizia )
		{
			indexEmergenza = i;
			break;
		}

		double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

        P += std::max(0., -0.038 + m_beta * Tm);
	}

	m_numeroFoglieTotali = 6.5 + m_sigma * exp( - stazione.Fotoperiodo(indexEmergenza) / 4. ) + 0.65 * NumeroFoglieEmerse(P);
	
    sprintf(message, "indice:%d  doy:%d  fotoPeriodo:%6.3f \n", indexEmergenza, getDoyFromDate(stazione.getDate(indexEmergenza)), stazione.Fotoperiodo(indexEmergenza));
	console.Show(message);

	if( m_numeroFoglieTotali > 20. )
	{
		sprintf(message, "Exception raised: numero di foglie totali non calcolato\n");
		console.Show(message);
		sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
		console.Show(message);
		return false;
	}

	return true;
}

// calcolo della data di emergenza
void Cereali::Emergenza(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += std::max( static_cast<double>(0.), TassoSviluppoEmergenza(Tm) );
}

// calcolo del numero di bozze fogliari e del periodo di viraggio apicale
void Cereali::Viraggio(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

    double Pr = std::max(0., -0.038 + m_beta * Tm);

	m_numeroPrimordi += Pr;
	m_faseFenologica[i] += Pr / ( static_cast<double>(m_numeroFoglieTotali) - 4. );
	//double F = NumeroFoglieEmerse(m_numeroPrimordi);
	//m_faseFenologica[i] = 1 + F / floor(m_numeroFoglieTotali);
}


// calcolo della data di spigatura
void Cereali::Spigatura(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

    double Pr = std::max(0., -0.038 + m_beta * Tm);

	m_numeroPrimordi += Pr;
	m_faseFenologica[i] += Pr * ( 1. - 0.03 * NumeroFoglieEmerse(m_numeroPrimordi) ) / ( m_numeroFoglieTotali - NumeroFoglieEmerse(m_numeroPrimordi) );
	// m_faseFenologica[i] += Pr * ( 1. - m_alfa * NumeroFoglieEmerse(m_numeroPrimordi) ) / ( m_numeroFoglieTotali + .75 - NumeroFoglieEmerse(m_numeroPrimordi) );
}


// calcolo della data di maturazione fisiologica
void Cereali::Maturazione(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += std::max( static_cast<double>(0.), ( Tm - m_sogliaGradiGiorno ) / m_limiteGradiGiorno );
}


// simulazione dello sviluppo delle fasi fenologiche dei cereali
void Cereali::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console)
{
	char message[256];

	if( stazione.NumeroGiorni() < 5 ) 
	{
		sprintf(message, "Exception raised: dati giornalieri non sufficienti\n");
		console.Show(message);
		sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
		console.Show(message);
		return;
	}

	if( m_faseFenologica.empty() )
		m_faseFenologica.push_back( static_cast<double>(0.) );
	else
		m_faseFenologica.push_back( m_faseFenologica[m_faseFenologica.size()-1] );

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
				if( ! CalcoloDateFittizie(stazione, parametri, console) )
					m_faseFenologica[m_faseFenologica.size()-1] = parametri.dato_mancante;
			}

			break;

		case 1:
			Emergenza(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
                m_emergenza = stazione.getDate(m_faseFenologica.size()-1);
				if( ! CalcoloNumeroFoglie(stazione, parametri, console) )
					m_faseFenologica[m_faseFenologica.size()-1] = parametri.dato_mancante;
			}

			break;

		case 2:
			Viraggio(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
                m_viraggioApicale = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Spigatura(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4.)
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
                m_spigatura = stazione.getDate(m_faseFenologica.size()-1);
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


double Cereali::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
{ 
	double bbch = m_faseFenologica[giorno];

	if( parametri.scalaBBCH )
	{
		switch( static_cast<long>( floor(m_faseFenologica[giorno]) ) )
		{
			case 1:
				if( parametri.coltura == 1 ) 
					// frumento duro
					bbch = 9. * ( m_faseFenologica[giorno] - 1. );
				else 
					// frumento tenero o orzo
					bbch = 10. * ( m_faseFenologica[giorno] - 1. );
			
				break;

			case 2:
				if( parametri.coltura == 1 ) 
				{
					// frumento duro
					bbch = 9. + ( 30. - 9. ) * (m_faseFenologica[giorno] - 2.);
				}
				else 
				{
					// frumento tenero o orzo
					bbch = 10. + ( 30. - 10. ) * (m_faseFenologica[giorno] - 2.);	
				}
				
				break;

			case 3:
				if( parametri.coltura == 0 || parametri.coltura == 1 ) 
					// frumento tenero o duro
					bbch = 30. + ( 59. - 30. ) * ( m_faseFenologica[giorno] - 3. );
				else 
					// orzo
					bbch = 30. + ( 65. - 30. ) * ( m_faseFenologica[giorno] - 3. );

				break;

			case 4:
				if( parametri.coltura == 0 ) 
					// frumento tenero
					bbch = 59. + ( 87. - 59. ) * ( m_faseFenologica[giorno] - 4. );
				else 
					if( parametri.coltura == 1 ) 
						// frumento duro
						bbch = 59. + ( 85. - 59. ) * ( m_faseFenologica[giorno] - 4. );
					else 
						// orzo
						bbch = 65. + ( 92. - 65. ) * ( m_faseFenologica[giorno] - 4. );

				break;

			case 5:
				if( parametri.coltura == 0 ) 
					// frumento tenero
					bbch = 87.;
				else 
					if( parametri.coltura == 1 ) 
						// frumento duro
						bbch = 85.;
					else 
						// orzo
						bbch = 92.;

				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}

