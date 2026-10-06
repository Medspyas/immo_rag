import { useEffect, useState } from 'react'

import { api } from '@/lib/api'

export type EtatService = 'verification' | 'ok' | 'base-injoignable' | 'hors-ligne'

const INTERVALLE_MS = 30_000

// État du backend, revérifié toutes les 30 s : l'utilisateur sait avant de
// poser sa question si le service peut répondre.
export function useSante(): EtatService {
  const [etat, setEtat] = useState<EtatService>('verification')

  useEffect(() => {
    let actif = true
    const verifier = async () => {
      try {
        const sante = await api.sante()
        if (actif) setEtat(sante.db === 'ok' ? 'ok' : 'base-injoignable')
      } catch {
        if (actif) setEtat('hors-ligne')
      }
    }
    void verifier()
    const minuteur = setInterval(verifier, INTERVALLE_MS)
    return () => {
      actif = false
      clearInterval(minuteur)
    }
  }, [])

  return etat
}
