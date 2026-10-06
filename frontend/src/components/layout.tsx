import { Library, MessageSquareText, Moon, Scale, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { NavLink, Outlet } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { type EtatService, useSante } from '@/hooks/use-sante'
import { cn } from '@/lib/utils'

const NAVIGATION = [
  { vers: '/', libelle: 'Assistant', icone: MessageSquareText },
  { vers: '/documents', libelle: 'Textes indexés', icone: Library },
]

const LIBELLE_ETAT: Record<EtatService, string> = {
  verification: 'Vérification…',
  ok: 'Service opérationnel',
  'base-injoignable': 'Base de données injoignable',
  'hors-ligne': 'Serveur injoignable',
}

function Marque() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <Scale className="size-4" aria-hidden />
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-semibold tracking-tight">immo-rag</span>
        <span className="block text-xs text-muted-foreground">Droit immobilier</span>
      </span>
    </div>
  )
}

function IndicateurSante() {
  const etat = useSante()
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
      <span
        aria-hidden
        className={cn(
          'size-2 rounded-full',
          etat === 'ok' && 'bg-emerald-500',
          etat === 'verification' && 'bg-muted-foreground/40',
          (etat === 'base-injoignable' || etat === 'hors-ligne') && 'bg-destructive',
        )}
      />
      {LIBELLE_ETAT[etat]}
    </div>
  )
}

function BasculeTheme() {
  const { resolvedTheme, setTheme } = useTheme()
  const sombre = resolvedTheme === 'dark'
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(sombre ? 'light' : 'dark')}
      aria-label={sombre ? 'Passer en thème clair' : 'Passer en thème sombre'}
    >
      {sombre ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  )
}

function LienNavigation({ vers, libelle, icone: Icone }: (typeof NAVIGATION)[number]) {
  return (
    <NavLink
      to={vers}
      end
      className={({ isActive }) =>
        cn(
          'flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors',
          'focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none',
          isActive
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
        )
      }
    >
      <Icone className="size-4" aria-hidden />
      {libelle}
    </NavLink>
  )
}

// Barre latérale sur grand écran, barre du haut sur mobile ; la page active
// s'affiche à droite (Outlet).
export function Layout() {
  return (
    <div className="flex min-h-dvh flex-col md:flex-row">
      <a
        href="#contenu"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-background focus:px-3 focus:py-2"
      >
        Aller au contenu
      </a>

      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r bg-sidebar p-4 md:flex">
        <Marque />
        <nav className="mt-8 flex flex-col gap-1" aria-label="Navigation principale">
          {NAVIGATION.map((n) => (
            <LienNavigation key={n.vers} {...n} />
          ))}
        </nav>
        <div className="mt-auto flex items-center justify-between gap-2 pt-4">
          <IndicateurSante />
          <BasculeTheme />
        </div>
      </aside>

      <header className="sticky top-0 z-20 border-b bg-background/90 backdrop-blur md:hidden">
        <div className="flex items-center justify-between px-4 py-3">
          <Marque />
          <BasculeTheme />
        </div>
        <nav className="flex gap-1 px-2 pb-2" aria-label="Navigation principale">
          {NAVIGATION.map((n) => (
            <LienNavigation key={n.vers} {...n} />
          ))}
        </nav>
      </header>

      <main id="contenu" className="flex min-w-0 flex-1 flex-col">
        <Outlet />
      </main>
    </div>
  )
}
