import { PageHeader } from "@/components/layout/PageHeader";
import { MethodologyAlert } from "@/components/feedback/MethodologyAlert";
import { FEATURE_GROUPS, TARGET_DEFINITIONS, TIME_SEGMENTS } from "@/lib/constants";
import { formatDate } from "@/lib/formatters";

const SECTIONS = [
  { id: "progetto", label: "Il progetto" },
  { id: "fonti", label: "Fonti NASA FIRMS" },
  { id: "rilevamento-vs-incendio", label: "Rilevamento vs incendio reale" },
  { id: "target", label: "Target fire_next_7d e fire_count_next_7d" },
  { id: "griglia", label: "Griglia geografica" },
  { id: "feature", label: "Lag, finestre mobili e feature" },
  { id: "frp", label: "Fire Radiative Power (FRP)" },
  { id: "segmentazione", label: "Segmentazione temporale" },
  { id: "split", label: "Train, validation, embargo, test" },
  { id: "limiti", label: "Limiti metodologici" },
  { id: "glossario", label: "Glossario" },
];

export const metadata = { title: "Documentazione" };

export default function DocumentationPage() {
  return (
    <div className="grid gap-6 lg:grid-cols-[15rem_minmax(0,1fr)]">
      <nav aria-label="Indice della documentazione" className="hidden h-fit lg:sticky lg:top-28 lg:block">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.1em] text-muted">Indice</p>
        <ul className="space-y-1 border-l text-sm">
          {SECTIONS.map((section) => (
            <li key={section.id}>
              <a href={`#${section.id}`} className="block border-l-2 border-transparent py-1.5 pl-3 text-muted hover:border-brand-400 hover:text-ink">
                {section.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <div className="min-w-0 space-y-10">
        <PageHeader
          title="Documentazione"
          description="Guida metodologica al progetto SpaceRiskIntelligence: fonti, definizioni, feature e limiti del prototipo di ricerca."
        />

        <details className="rounded-2xl border bg-surface p-4 shadow-panel lg:hidden">
          <summary className="cursor-pointer font-semibold">Indice della documentazione</summary>
          <ul className="mt-3 space-y-1 text-sm">
            {SECTIONS.map((section) => (
              <li key={section.id}>
                <a href={`#${section.id}`} className="text-brand-300 hover:underline">
                  {section.label}
                </a>
              </li>
            ))}
          </ul>
        </details>

        <Section id="progetto" title="Il progetto">
          <p>
            SpaceRiskIntelligence è un prototipo di ricerca che analizza i rilevamenti satellitari NASA FIRMS su scala globale per
            stimare la probabilità sperimentale di ulteriore attività rilevata nelle celle geografiche nei sette giorni successivi.
            Il progetto affronta due problemi correlati: la classificazione binaria di <code>fire_next_7d</code> e la stima
            sperimentale di <code>fire_count_next_7d</code>.
          </p>
        </Section>

        <Section id="fonti" title="Fonti NASA FIRMS">
          <p>
            I dati derivano dal Fire Information for Resource Management System (FIRMS) della NASA, che distribuisce i
            rilevamenti satellitari di anomalie termiche raccolti da più sensori. Il progetto consolida più estrazioni di questa
            fonte in un unico dataset grezzo globale prima della segmentazione temporale e del campionamento.
          </p>
        </Section>

        <Section id="rilevamento-vs-incendio" title="Rilevamento vs incendio reale">
          <p>
            Un <strong>rilevamento satellitare</strong> è un&apos;anomalia termica identificata da un sensore in un dato momento e
            in una data posizione: non equivale necessariamente a un incendio fisico distinto. Più rilevamenti ravvicinati nello
            spazio e nel tempo possono derivare dallo stesso evento; viceversa, incendi di bassa intensità o coperti da nuvole
            possono non generare alcun rilevamento. L&apos;interfaccia evita quindi di parlare di &quot;numero certo di incendi
            futuri&quot;, preferendo &quot;rilevamenti&quot; e &quot;attività rilevata&quot;.
          </p>
        </Section>

        <Section id="target" title="Target fire_next_7d e fire_count_next_7d">
          <div className="space-y-3">
            {TARGET_DEFINITIONS.map((target) => (
              <div key={target.name} className="rounded-xl border bg-elevated p-4">
                <p className="font-mono text-sm font-semibold">{target.name}</p>
                <p className="text-xs uppercase tracking-wide text-brand-300">{target.kind}</p>
                <p className="mt-2 text-sm leading-6 text-muted">{target.description}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="griglia" title="Griglia geografica">
          <p>
            La superficie terrestre viene discretizzata in celle di 0,1° di lato (circa 11 km all&apos;equatore). Il campionamento
            seleziona un sottoinsieme di celle storicamente associate ad attività, per un totale di 15.000 celle (5.000 per
            segmento temporale utilizzabile). Questo rende il campione condizionato e non rappresentativo di una selezione
            casuale della superficie terrestre.
          </p>
        </Section>

        <Section id="feature" title="Lag, finestre mobili e feature">
          <p className="mb-4">
            Le 17 feature del modello sono organizzate in cinque famiglie. Tutte le finestre mobili escludono il giorno corrente
            tramite uno shift temporale di un giorno, per evitare che il modello osservi informazioni non ancora disponibili al
            momento della previsione.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            {FEATURE_GROUPS.map((group) => (
              <div key={group.id} className="rounded-xl border bg-elevated p-4">
                <p className="font-semibold text-brand-300">{group.label}</p>
                <p className="mt-1 text-xs text-muted">{group.description}</p>
                <ul className="mt-2 space-y-1 text-sm">
                  {group.features.map((feature) => (
                    <li key={feature.name}>
                      <code className="text-xs">{feature.name}</code> — <span className="text-muted">{feature.description}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Section>

        <Section id="frp" title="Fire Radiative Power (FRP)">
          <p>
            La Fire Radiative Power (FRP) misura l&apos;intensità radiativa istantanea rilevata dal sensore satellitare, espressa
            in megawatt (MW). Le feature basate su FRP aggregano questa intensità su finestre di 7 e 14 giorni, fornendo
            un&apos;indicazione dell&apos;intensità dell&apos;attività rilevata oltre alla sua semplice presenza/assenza.
          </p>
        </Section>

        <Section id="segmentazione" title="Segmentazione temporale">
          <p className="mb-4">
            Il periodo osservato non è continuo: la disponibilità dei dati grezzi presenta interruzioni. La pipeline individua
            segmenti temporali continui e utilizza solo quelli di almeno 28 giorni per addestrare e valutare i modelli.
          </p>
          <div className="overflow-x-auto rounded-xl border">
            <table className="w-full min-w-[480px] border-collapse text-sm">
              <thead>
                <tr className="border-b bg-elevated text-left text-xs uppercase tracking-wide text-muted">
                  <th scope="col" className="px-3 py-2">
                    Segmento
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Periodo
                  </th>
                  <th scope="col" className="px-3 py-2 text-right">
                    Giorni
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Utilizzabile
                  </th>
                </tr>
              </thead>
              <tbody>
                {TIME_SEGMENTS.map((segment) => (
                  <tr key={segment.id} className="border-b last:border-0">
                    <td className="px-3 py-2 font-medium">Segmento {segment.id}</td>
                    <td className="px-3 py-2 text-muted">
                      {formatDate(segment.start)} – {formatDate(segment.end)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{segment.days}</td>
                    <td className="px-3 py-2">{segment.usableForModel ? "Sì" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            Il segmento 5 è continuo ma troppo corto per produrre righe nel dataset ML: non dispone di storia sufficiente per le
            feature né di un orizzonte futuro completo di 7 giorni.
          </p>
        </Section>

        <Section id="split" title="Train, validation, embargo, test">
          <p>
            All&apos;interno di ciascun segmento utilizzabile, i dati vengono divisi in ordine cronologico in train, validation e
            test, con una finestra di <strong>embargo di 7 giorni</strong> fra ciascuna porzione. L&apos;embargo evita che
            informazioni sovrapposte alle finestre mobili delle feature attraversino il confine fra le porzioni, riducendo il
            rischio di ottimismo nella valutazione. La soglia di decisione è scelta esclusivamente sul set di validation, mai
            osservando il test.
          </p>
        </Section>

        <Section id="limiti" title="Limiti metodologici">
          <ul className="list-disc space-y-2 pl-5 text-sm leading-6 text-muted">
            <li>Le metriche derivano da un singolo split temporale isolato (segmento 0), non da una validazione multi-segmento.</li>
            <li>Il campione di celle è condizionato alla storicità di attività, non casuale.</li>
            <li>La generalizzazione geografica a regioni non rappresentate nel campione non è stata verificata.</li>
            <li>Nessun modello è stato serializzato al termine del run: le metriche derivano da un&apos;esecuzione singola.</li>
            <li>La regressione di fire_count_next_7d è sperimentale e non è stata sottoposta allo stesso livello di validazione della classificazione.</li>
          </ul>
        </Section>

        <Section id="glossario" title="Glossario">
          <dl className="space-y-3 text-sm">
            {[
              ["FIRMS", "Fire Information for Resource Management System, il sistema NASA che distribuisce i rilevamenti satellitari."],
              ["Rilevamento", "Anomalia termica identificata da un sensore satellitare in un punto e in un istante specifici."],
              ["FRP", "Fire Radiative Power: intensità radiativa istantanea rilevata dal sensore, in megawatt."],
              ["Cella geografica", "Porzione di superficie terrestre di 0,1° di lato usata come unità spaziale di analisi."],
              ["Lag", "Presenza/assenza di rilevamento a una distanza temporale fissa nel passato."],
              ["Finestra mobile", "Intervallo di giorni precedenti (esclusi il giorno corrente) su cui si aggregano le feature."],
              ["Embargo", "Intervallo temporale escluso ai confini fra train, validation e test per evitare fughe di informazione."],
              ["Soglia di decisione", "Valore di probabilità sopra il quale una previsione viene classificata come positiva."],
              ["ROC-AUC / PR-AUC", "Aree sotto le curve ROC e Precision-Recall, sintetizzano la capacità di separare le classi."],
            ].map(([term, definition]) => (
              <div key={term}>
                <dt className="font-semibold">{term}</dt>
                <dd className="text-muted">{definition}</dd>
              </div>
            ))}
          </dl>
        </Section>

        <MethodologyAlert
          title="Promemoria"
          items={["Questa documentazione descrive un prototipo di ricerca in evoluzione: definizioni e soglie possono cambiare nelle prossime iterazioni."]}
        />
      </div>
    </div>
  );
}

function Section({ id, title, children }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-28">
      <h2 id={`${id}-title`} className="mb-3 text-xl font-bold tracking-tight">
        {title}
      </h2>
      <div className="text-sm leading-6 text-muted [&_strong]:text-ink [&_code]:rounded [&_code]:bg-elevated [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-ink">
        {children}
      </div>
    </section>
  );
}
