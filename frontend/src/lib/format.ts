// Mise en forme des données du backend pour l'affichage.

// Titre court d'un texte de loi : les intitulés officiels sont très longs
// (« Loi n° 89-462 du 6 juillet 1989 tendant à améliorer… »). On garde la
// partie jusqu'à l'année ; un texte sans date (« Code civil ») reste entier.
export function titreCourt(titre: string | null | undefined, repli: string): string {
  if (!titre) return repli
  const jusquAnnee = titre.match(/^.*?\b(1[89]|20)\d{2}\b/)
  return jusquAnnee ? jusquAnnee[0] : titre
}

// Suite de l'intitulé officiel après le titre court (« tendant à améliorer
// les rapports locatifs… »), affichée en sous-titre ; vide s'il n'y en a pas.
export function suiteTitre(titre: string | null | undefined, repli: string): string {
  if (!titre) return ''
  return titre.slice(titreCourt(titre, repli).length).trim()
}

const formatDate = new Intl.DateTimeFormat('fr-FR', { dateStyle: 'long' })

export function dateLongue(iso: string): string {
  return formatDate.format(new Date(iso))
}

const formatNombre = new Intl.NumberFormat('fr-FR')

export function nombre(n: number): string {
  return formatNombre.format(n)
}
