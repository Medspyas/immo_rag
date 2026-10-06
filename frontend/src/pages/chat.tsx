import { ArrowRight, Quote, ShieldCheck, ShieldOff } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Composer } from '@/components/chat/composer'
import { Attente, Erreur, Refus, ReponseFondee, type Titres } from '@/components/chat/reponse'
import { useDocuments } from '@/hooks/use-documents'
import { api, type ReponseGeneree } from '@/lib/api'

interface Echange {
  id: string
  question: string
  reponse?: ReponseGeneree
  erreur?: string
}

// Questions d'exemple, tirées du jeu d'évaluation (eval/golden_dataset.jsonl).
const EXEMPLES = [
  'Quel délai de préavis le bailleur doit-il respecter pour donner congé à son locataire ?',
  'Dans quel délai le dépôt de garantie doit-il être restitué au locataire ?',
  'Quelles sont les missions du syndic de copropriété ?',
  "Quelles activités relèvent de la loi Hoguet ?",
]

const GARANTIES = [
  { icone: Quote, texte: 'Chaque réponse cite le passage exact de la loi' },
  { icone: ShieldCheck, texte: 'Les citations sont vérifiées dans le texte source' },
  { icone: ShieldOff, texte: "Sans source fiable, l'assistant s'abstient" },
]

// La conversation survit à un passage sur la page « Textes indexés » et à un
// rechargement de l'onglet, pas à sa fermeture. Le stockage peut être
// indisponible (navigation privée) : l'application fonctionne alors sans.
const CLE_STOCKAGE = 'immo-rag:conversation'

function lireConversation(): Echange[] {
  try {
    const brut = sessionStorage.getItem(CLE_STOCKAGE)
    const echanges: Echange[] = brut ? JSON.parse(brut) : []
    // Une question restée sans réponse (onglet rechargé pendant l'attente)
    // ne recevra jamais la sienne : on la retire.
    return echanges.filter((e) => e.reponse || e.erreur)
  } catch {
    return []
  }
}

function Accueil({ choisir }: { choisir: (q: string) => void }) {
  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col items-center py-10 text-center md:py-16">
      <h1 className="text-3xl font-semibold tracking-tight text-balance md:text-4xl">
        Une question de droit immobilier ?
      </h1>
      <p className="mt-3 max-w-xl text-muted-foreground text-pretty">
        Bail d'habitation, copropriété, transaction, fiscalité : réponses fondées sur les textes consolidés de
        Légifrance.
      </p>
      <ul className="mt-6 flex flex-wrap justify-center gap-x-5 gap-y-2 text-sm text-muted-foreground">
        {GARANTIES.map(({ icone: Icone, texte }) => (
          <li key={texte} className="flex items-center gap-1.5">
            <Icone className="size-4 text-primary" aria-hidden />
            {texte}
          </li>
        ))}
      </ul>
      <div className="mt-10 grid w-full gap-2 sm:grid-cols-2">
        {EXEMPLES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => choisir(q)}
            className="group flex items-start justify-between gap-3 rounded-lg border bg-card p-4 text-left text-sm transition-colors hover:border-primary/40 hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
          >
            {q}
            <ArrowRight
              className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-primary motion-reduce:transition-none"
              aria-hidden
            />
          </button>
        ))}
      </div>
    </div>
  )
}

export function PageChat() {
  const [echanges, setEchanges] = useState<Echange[]>(lireConversation)
  const { documents } = useDocuments()
  const fin = useRef<HTMLDivElement>(null)

  const titres = useMemo<Titres>(
    () => Object.fromEntries((documents ?? []).map((d) => [d.doc_id, d.titre])),
    [documents],
  )
  const occupe = echanges.some((e) => !e.reponse && !e.erreur)

  useEffect(() => {
    try {
      sessionStorage.setItem(CLE_STOCKAGE, JSON.stringify(echanges))
    } catch {
      // stockage indisponible : la conversation reste en mémoire seulement
    }
    fin.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [echanges])

  const majEchange = (id: string, maj: Partial<Echange>) =>
    setEchanges((liste) => liste.map((e) => (e.id === id ? { ...e, ...maj } : e)))

  const interroger = async (id: string, question: string) => {
    try {
      majEchange(id, { reponse: await api.poser(question) })
    } catch (e) {
      majEchange(id, { erreur: e instanceof Error ? e.message : String(e) })
    }
  }

  const envoyer = (question: string) => {
    const id = crypto.randomUUID()
    setEchanges((liste) => [...liste, { id, question }])
    void interroger(id, question)
  }

  const reessayer = (echange: Echange) => {
    majEchange(echange.id, { erreur: undefined })
    void interroger(echange.id, echange.question)
  }

  return (
    <div className="flex flex-1 flex-col">
      <div className="flex-1 px-4 md:px-8">
        {echanges.length === 0 ? (
          <Accueil choisir={envoyer} />
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-8">
            {echanges.map((e) => (
              <div key={e.id} className="flex flex-col gap-3">
                <p className="max-w-[85%] self-end rounded-2xl rounded-br-sm bg-secondary px-4 py-2.5 text-secondary-foreground">
                  {e.question}
                </p>
                {e.erreur ? (
                  <Erreur message={e.erreur} reessayer={() => reessayer(e)} />
                ) : !e.reponse ? (
                  <Attente />
                ) : e.reponse.refus ? (
                  <Refus motif={e.reponse.motif} />
                ) : (
                  <ReponseFondee reponse={e.reponse} titres={titres} />
                )}
              </div>
            ))}
            <div ref={fin} />
          </div>
        )}
      </div>

      <div className="sticky bottom-0 bg-gradient-to-t from-background via-background to-background/0 px-4 pt-6 pb-4 md:px-8">
        <div className="mx-auto w-full max-w-3xl">
          <Composer envoyer={envoyer} occupe={occupe} />
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Outil d'aide à la recherche : ne remplace pas l'avis d'un professionnel du droit.
          </p>
        </div>
      </div>
    </div>
  )
}
