import { useCallback, useEffect, useState } from 'react'

import { api, type DocumentIndexe } from '@/lib/api'

type Etat = { documents: DocumentIndexe[] | null; erreur: string | null }

async function charger(): Promise<Etat> {
  try {
    return { documents: await api.documents(), erreur: null }
  } catch (e) {
    return { documents: null, erreur: e instanceof Error ? e.message : String(e) }
  }
}

// Liste des textes indexés, rechargeable après une suppression. Sert aussi au
// chat pour afficher le titre d'une loi citée au lieu de son identifiant.
export function useDocuments() {
  const [etat, setEtat] = useState<Etat>({ documents: null, erreur: null })

  useEffect(() => {
    let actif = true
    void charger().then((e) => actif && setEtat(e))
    return () => {
      actif = false
    }
  }, [])

  const recharger = useCallback(async () => {
    setEtat(await charger())
  }, [])

  return { ...etat, recharger }
}
