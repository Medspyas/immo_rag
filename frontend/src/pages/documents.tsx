import { AlertCircle, FileText, RotateCcw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { useDocuments } from '@/hooks/use-documents'
import { api, type DocumentIndexe } from '@/lib/api'
import { dateLongue, nombre, suiteTitre, titreCourt } from '@/lib/format'

function Chiffre({ valeur, libelle }: { valeur: string; libelle: string }) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="text-2xl font-semibold tracking-tight tabular-nums">{valeur}</div>
      <div className="mt-0.5 text-sm text-muted-foreground">{libelle}</div>
    </div>
  )
}

// Suppression irréversible : confirmation obligatoire, bouton rouge à part.
function BoutonSupprimer({ document, apres }: { document: DocumentIndexe; apres: () => void }) {
  const [enCours, setEnCours] = useState(false)
  const nom = titreCourt(document.titre, document.doc_id)

  const supprimer = async () => {
    setEnCours(true)
    try {
      await api.supprimer(document.doc_id)
      toast.success(`${nom} retiré de la base.`)
      apres()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e))
    } finally {
      setEnCours(false)
    }
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          disabled={enCours}
          aria-label={`Retirer ${nom}`}
          className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
        >
          <Trash2 className="size-4" />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Retirer {nom} ?</AlertDialogTitle>
          <AlertDialogDescription>
            Ses {nombre(document.morceaux)} articles seront supprimés de la base : l'assistant ne
            pourra plus s'appuyer sur ce texte. Il faudra le réindexer pour le retrouver.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Annuler</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={supprimer}>
            Retirer le texte
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function Chargement() {
  return (
    <div className="space-y-2" aria-busy="true" aria-label="Chargement des textes">
      {Array.from({ length: 5 }, (_, i) => (
        <Skeleton key={i} className="h-12 w-full" />
      ))}
    </div>
  )
}

export function PageDocuments() {
  const { documents, erreur, recharger } = useDocuments()
  const total = documents?.reduce((s, d) => s + d.morceaux, 0) ?? 0

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-8 md:px-8 md:py-12">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Textes indexés</h1>
        <p className="mt-1 max-w-2xl text-muted-foreground">
          Les textes dans lesquels l'assistant cherche ses réponses. Seules les versions consolidées,
          en vigueur, sont indexées, découpées article par article.
        </p>
      </header>

      {erreur ? (
        <div
          role="alert"
          className="mt-8 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm"
        >
          <span className="flex items-center gap-2">
            <AlertCircle className="size-4 text-destructive" aria-hidden />
            {erreur}
          </span>
          <Button variant="outline" size="sm" onClick={recharger}>
            <RotateCcw className="size-4" />
            Réessayer
          </Button>
        </div>
      ) : documents === null ? (
        <div className="mt-8">
          <Chargement />
        </div>
      ) : documents.length === 0 ? (
        <div className="mt-8 flex flex-col items-center rounded-xl border border-dashed p-10 text-center">
          <FileText className="size-8 text-muted-foreground" aria-hidden />
          <p className="mt-3 font-medium">Aucun texte indexé</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Lancez <code className="rounded bg-muted px-1.5 py-0.5">python scripts/ingerer_corpus.py</code>{' '}
            pour indexer le corpus.
          </p>
        </div>
      ) : (
        <>
          <div className="mt-8 grid grid-cols-2 gap-3 sm:max-w-md">
            <Chiffre valeur={nombre(documents.length)} libelle="textes" />
            <Chiffre valeur={nombre(total)} libelle="articles indexés" />
          </div>

          <div className="mt-6 overflow-hidden rounded-xl border bg-card">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-full pl-4">Texte</TableHead>
                  <TableHead className="text-right">Articles</TableHead>
                  <TableHead className="hidden whitespace-nowrap sm:table-cell">Indexé le</TableHead>
                  <TableHead className="w-12">
                    <span className="sr-only">Actions</span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents.map((d) => (
                  <TableRow key={d.doc_id}>
                    <TableCell className="pl-4 whitespace-normal">
                      <div className="font-medium">{titreCourt(d.titre, d.doc_id)}</div>
                      {suiteTitre(d.titre, d.doc_id) && (
                        <div className="line-clamp-1 text-xs text-muted-foreground" title={d.titre ?? ''}>
                          {suiteTitre(d.titre, d.doc_id)}
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{nombre(d.morceaux)}</TableCell>
                    <TableCell className="hidden whitespace-nowrap text-muted-foreground sm:table-cell">
                      {dateLongue(d.mis_a_jour)}
                    </TableCell>
                    <TableCell>
                      <BoutonSupprimer document={d} apres={recharger} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  )
}
