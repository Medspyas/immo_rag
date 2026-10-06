import { AlertCircle, Check, Copy, Loader2, RotateCcw, ShieldCheck, ShieldOff } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import type { Citation, ReponseGeneree } from '@/lib/api'
import { titreCourt } from '@/lib/format'

// Titre affiché pour un identifiant de document (doc_id → titre complet).
export type Titres = Record<string, string | null>

function CitationSource({ citation, titres }: { citation: Citation; titres: Titres }) {
  return (
    <figure className="rounded-lg border bg-muted/40 p-4">
      <figcaption className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline" className="border-primary/30 font-semibold text-primary">
          Art. {citation.article}
        </Badge>
        <span>{titreCourt(titres[citation.doc_id], citation.doc_id)}</span>
      </figcaption>
      <blockquote className="border-l-2 border-primary pl-3 font-serif text-[1.05rem] leading-relaxed italic">
        « {citation.passage} »
      </blockquote>
    </figure>
  )
}

// Texte prêt à coller dans un courriel : la réponse puis ses sources.
function versTexte(reponse: ReponseGeneree, titres: Titres): string {
  const sources = reponse.citations.map(
    (c) => `- ${titreCourt(titres[c.doc_id], c.doc_id)}, art. ${c.article} : « ${c.passage} »`,
  )
  return `${reponse.reponse}\n\nSources :\n${sources.join('\n')}`
}

function BoutonCopier({ texte }: { texte: string }) {
  const [copie, setCopie] = useState(false)
  const copier = async () => {
    try {
      await navigator.clipboard.writeText(texte)
      setCopie(true)
      setTimeout(() => setCopie(false), 2000)
    } catch {
      toast.error('Copie impossible dans ce navigateur.')
    }
  }
  return (
    <Button variant="ghost" size="sm" onClick={copier} className="text-muted-foreground">
      {copie ? <Check className="size-4" /> : <Copy className="size-4" />}
      {copie ? 'Copié' : 'Copier avec les sources'}
    </Button>
  )
}

export function ReponseFondee({ reponse, titres }: { reponse: ReponseGeneree; titres: Titres }) {
  return (
    <article className="rounded-xl border bg-card p-5 shadow-xs">
      <div className="mb-3 flex items-center gap-2 text-xs font-medium text-emerald-700 dark:text-emerald-400">
        <ShieldCheck className="size-4" aria-hidden />
        Réponse vérifiée : chaque citation a été retrouvée dans le texte de loi
      </div>
      <p className="leading-relaxed whitespace-pre-line">{reponse.reponse}</p>

      {reponse.citations.length > 0 && (
        <section className="mt-5" aria-label="Sources citées">
          <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            Sources
          </h3>
          <div className="flex flex-col gap-2">
            {reponse.citations.map((c, i) => (
              <CitationSource key={`${c.doc_id}-${c.article}-${i}`} citation={c} titres={titres} />
            ))}
          </div>
        </section>
      )}

      <div className="mt-3 flex justify-end">
        <BoutonCopier texte={versTexte(reponse, titres)} />
      </div>
    </article>
  )
}

// Un refus n'est pas une panne : c'est le garde-fou qui fonctionne. Le ton est
// neutre, et le motif explique pourquoi aucune réponse n'est donnée.
export function Refus({ motif }: { motif: string | null }) {
  return (
    <article className="rounded-xl border border-dashed bg-muted/30 p-5">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <ShieldOff className="size-4 text-muted-foreground" aria-hidden />
        Pas de réponse fondée dans les textes indexés
      </div>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
        Les textes disponibles ne permettent pas de répondre avec certitude. L'assistant préfère
        s'abstenir plutôt que d'avancer une réponse non vérifiée.
      </p>
      {motif && (
        <p className="mt-2 text-xs text-muted-foreground">
          <span className="font-medium">Motif du garde-fou :</span> {motif}
        </p>
      )}
      <p className="mt-3 text-xs text-muted-foreground">
        Reformulez la question avec les termes juridiques, ou consultez un professionnel du droit.
      </p>
    </article>
  )
}

export function Attente() {
  return (
    <div className="rounded-xl border bg-card p-5" aria-live="polite">
      <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" aria-hidden />
        Recherche dans les textes, rédaction puis vérification des citations…
      </div>
      <div className="space-y-2">
        <Skeleton className="h-4 w-11/12" />
        <Skeleton className="h-4 w-4/5" />
        <Skeleton className="h-16 w-full" />
      </div>
    </div>
  )
}

export function Erreur({ message, reessayer }: { message: string; reessayer: () => void }) {
  return (
    <div
      role="alert"
      className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm"
    >
      <span className="flex items-center gap-2">
        <AlertCircle className="size-4 text-destructive" aria-hidden />
        {message}
      </span>
      <Button variant="outline" size="sm" onClick={reessayer}>
        <RotateCcw className="size-4" />
        Réessayer
      </Button>
    </div>
  )
}
