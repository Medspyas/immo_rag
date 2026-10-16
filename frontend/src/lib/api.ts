// Client de l'API FastAPI. Les types reprennent les schémas Pydantic du
// backend (app/rag/generator.py, app/rag/vector_store.py) : un champ renommé
// côté Python doit l'être ici aussi.

// En développement, Vite relaie /api vers le port 8000 (vite.config.ts) ; en
// production, le serveur web fait de même (frontend/nginx.conf).
const BASE = import.meta.env.VITE_API_URL ?? '/api'

export interface Citation {
  doc_id: string
  article: string
  passage: string
}

export interface ReponseGeneree {
  refus: boolean
  motif: string | null
  reponse: string
  citations: Citation[]
  // Question réellement traitée quand une question de suivi a été réécrite
  // à partir de la conversation (app/rag/suivi.py) ; null sinon.
  question_comprise: string | null
}

// Un échange précédent, transmis pour comprendre une question de suivi.
export interface EchangePrecedent {
  question: string
  reponse: string
}

// Nombre d'échanges précédents acceptés par l'API (ECHANGES_MAX, suivi.py).
export const ECHANGES_MAX = 3

export interface DocumentIndexe {
  doc_id: string
  titre: string | null
  morceaux: number
  mis_a_jour: string
}

export interface Sante {
  status: string
  db: string
}

// Erreur porteuse du code HTTP : l'interface distingue un service saturé
// (503, réessayer plus tard) d'une question invalide (422).
export class ErreurApi extends Error {
  readonly statut: number

  constructor(statut: number, message: string) {
    super(message)
    this.statut = statut
  }
}

async function appeler<T>(chemin: string, init?: RequestInit): Promise<T> {
  let reponse: Response
  try {
    reponse = await fetch(`${BASE}${chemin}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new ErreurApi(0, "Impossible de joindre le serveur. Vérifiez qu'il est démarré.")
  }
  if (!reponse.ok) {
    throw new ErreurApi(reponse.status, await messageErreur(reponse))
  }
  if (reponse.status === 204) return undefined as T
  return reponse.json() as Promise<T>
}

// Message lisible pour l'utilisateur, selon le code renvoyé par FastAPI.
async function messageErreur(reponse: Response): Promise<string> {
  if (reponse.status === 503) {
    return 'Le service est momentanément saturé. Réessayez dans une minute.'
  }
  if (reponse.status === 422) {
    return 'Question invalide : elle doit contenir entre 1 et 1000 caractères.'
  }
  try {
    const corps = await reponse.json()
    if (typeof corps.detail === 'string') return corps.detail
  } catch {
    // corps non JSON : message générique ci-dessous
  }
  return `Erreur inattendue du serveur (${reponse.status}).`
}

export const api = {
  sante: () => appeler<Sante>('/health'),
  poser: (question: string, historique: EchangePrecedent[] = []) =>
    appeler<ReponseGeneree>('/query', {
      method: 'POST',
      body: JSON.stringify({ question, historique: historique.slice(-ECHANGES_MAX) }),
    }),
  documents: () => appeler<DocumentIndexe[]>('/documents'),
  supprimer: (docId: string) =>
    appeler<void>(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' }),
}
