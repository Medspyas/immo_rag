import { ArrowUp } from 'lucide-react'
import { type FormEvent, type KeyboardEvent, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

// Même plafond que le backend (app/models.py, QUESTION_MAX_CARACTERES).
export const QUESTION_MAX = 1000

interface Props {
  envoyer: (question: string) => void
  occupe: boolean
}

export function Composer({ envoyer, occupe }: Props) {
  const [texte, setTexte] = useState('')
  const question = texte.trim()
  const tropLong = texte.length > QUESTION_MAX
  const possible = question.length > 0 && !tropLong && !occupe

  const soumettre = (e?: FormEvent) => {
    e?.preventDefault()
    if (!possible) return
    envoyer(question)
    setTexte('')
  }

  // Entrée envoie, Maj+Entrée passe à la ligne (convention des messageries).
  const touche = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      soumettre()
    }
  }

  return (
    <form onSubmit={soumettre} className="relative rounded-xl border bg-card shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
      <label htmlFor="question" className="sr-only">
        Votre question
      </label>
      <Textarea
        id="question"
        value={texte}
        onChange={(e) => setTexte(e.target.value)}
        onKeyDown={touche}
        placeholder="Ex. : Dans quel délai le dépôt de garantie doit-il être restitué ?"
        rows={2}
        aria-describedby="aide-question"
        aria-invalid={tropLong}
        className="max-h-48 min-h-14 resize-none border-0 bg-transparent px-4 pt-3 pb-11 text-base shadow-none focus-visible:ring-0 dark:bg-transparent"
      />
      <div className="absolute inset-x-3 bottom-2.5 flex items-center justify-between gap-2">
        <span
          id="aide-question"
          className={cn('text-xs text-muted-foreground tabular-nums', tropLong ? 'text-destructive' : 'hidden sm:inline')}
        >
          {tropLong
            ? `${texte.length} / ${QUESTION_MAX} caractères : question trop longue`
            : 'Entrée pour envoyer, Maj+Entrée pour aller à la ligne'}
        </span>
        <Button type="submit" size="icon" disabled={!possible} aria-label="Envoyer la question" className="ml-auto">
          <ArrowUp className="size-4" />
        </Button>
      </div>
    </form>
  )
}
