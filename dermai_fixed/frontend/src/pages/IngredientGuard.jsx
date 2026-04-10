import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { flagIngredients, getProducts } from '../api'
import { SafeBadge, Spinner } from '../components/index.jsx'

const CATEGORIES = ['All', 'cleanser', 'moisturiser', 'SPF', 'serum', 'toner']
const SKIN_TYPES = ['All', 'oily', 'dry', 'combination', 'sensitive', 'normal']

const RISK_STYLES = {
  HIGH:     { card: 'bg-red-50 border-red-200',     badge: 'bg-red-500 text-white',    label: '🔴 HIGH' },
  MODERATE: { card: 'bg-amber-50 border-amber-200', badge: 'bg-amber-500 text-white',  label: '🟡 MODERATE' },
  LOW:      { card: 'bg-yellow-50 border-yellow-200', badge: 'bg-yellow-400 text-yellow-900', label: '🟠 LOW' },
}

const OVERALL_RISK_STYLES = {
  HIGH:     'bg-red-600 text-white',
  MODERATE: 'bg-amber-500 text-white',
  LOW:      'bg-yellow-400 text-yellow-900',
  SAFE:     'bg-green-600 text-white',
}

// Commonly harmful ingredients for quick paste
const EXAMPLE_INGREDIENTS = [
  'Aqua, Glycerin, Ceramide NP, Niacinamide, Panthenol, Allantoin',
  'Aqua, Alcohol Denat., Parfum, Methylparaben, Propylparaben, DMDM Hydantoin, Sodium Lauryl Sulfate',
  'Aqua, Zinc Oxide, Titanium Dioxide, Glycerin, Hyaluronic Acid, Tocopherol, Squalane',
]

function EnhancedIngredientCard({ ingredient, risk_level, explanation, category, ewg_score, sources }) {
  const [expanded, setExpanded] = useState(false)
  const style = RISK_STYLES[risk_level] || RISK_STYLES['LOW']
  return (
    <div className={`border rounded-xl p-3 text-sm ${style.card}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-semibold text-gray-800">{ingredient}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-bold ${style.badge}`}>{style.label}</span>
            {category && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-600">{category}</span>
            )}
            {ewg_score != null && (
              <span className="text-xs font-mono text-gray-500">EWG: {ewg_score}/10</span>
            )}
          </div>
          <p className="text-xs text-gray-700 leading-relaxed">{explanation}</p>
        </div>
        {sources?.length > 0 && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="text-xs text-gray-400 hover:text-gray-600 whitespace-nowrap ml-2"
          >
            {expanded ? '▲ less' : '▼ sources'}
          </button>
        )}
      </div>
      {expanded && sources?.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-200">
          <p className="text-xs font-medium text-gray-500 mb-1">References:</p>
          <ul className="space-y-0.5">
            {sources.map((s, i) => (
              <li key={i} className="text-xs text-gray-500">• {s}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function IngredientGuard() {
  const [inci, setInci]           = useState('')
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [products, setProducts]   = useState([])
  const [catFilter, setCatFilter] = useState('All')
  const [stFilter, setStFilter]   = useState('All')
  const [expanded, setExpanded]   = useState(null)
  const [prodLoading, setProdLoading] = useState(true)

  useEffect(() => {
    getProducts().then(r => setProducts(r.data.products)).catch(() => {}).finally(() => setProdLoading(false))
  }, [])

  const handleCheck = async () => {
    const list = inci.split(/[\n,]+/).map(s => s.trim()).filter(Boolean)
    if (!list.length) { toast.error('Paste an INCI list first.'); return }
    setLoading(true)
    try {
      const res = await flagIngredients(list)
      setResult(res.data)
      const { high_risk, moderate_risk } = res.data.summary || {}
      if (high_risk > 0) toast.error(`⚠️ ${high_risk} HIGH risk ingredient(s) found!`)
      else if (moderate_risk > 0) toast(`🟡 ${moderate_risk} moderate concern ingredient(s) found`, { icon: '⚠️' })
      else toast.success('✅ No major concerns found!')
    } catch {
      toast.error('Could not check ingredients. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const filtered = products.filter(p => {
    const catOk = catFilter === 'All' || p.category.toLowerCase() === catFilter.toLowerCase()
    const stOk  = stFilter === 'All' || p.skin_type_tags.map(s => s.toLowerCase()).includes(stFilter.toLowerCase())
    return catOk && stOk
  })

  const ewgColor = (score) =>
    score <= 3 ? 'text-green-600' : score <= 6 ? 'text-amber-600' : 'text-red-600'

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900">Ingredient Guard</h1>
        <p className="text-gray-500 mt-1">
          Check any product's INCI list against our EWG/SCCS/EU-backed hazard database (200+ ingredients).
        </p>
      </div>

      {/* Checker */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <h2 className="font-display font-semibold text-gray-800">Paste INCI Ingredient List</h2>
          <div className="flex gap-2 flex-wrap">
            {EXAMPLE_INGREDIENTS.map((ex, i) => (
              <button
                key={i}
                onClick={() => { setInci(ex); setResult(null) }}
                className="text-xs border border-teal-200 text-teal-600 hover:bg-teal-50 px-3 py-1 rounded-full transition-colors"
              >
                Example {i+1}
              </button>
            ))}
          </div>
        </div>
        <textarea
          value={inci}
          onChange={e => { setInci(e.target.value); setResult(null) }}
          rows={5}
          placeholder="Paste INCI list here — comma or newline separated. e.g. Aqua, Glycerin, Alcohol Denat., Parfum, Niacinamide, Ceramide NP, Methylparaben…"
          className="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 resize-none font-mono"
        />
        <div className="flex items-center gap-3">
          <button
            onClick={handleCheck}
            disabled={loading}
            className="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 text-white font-semibold px-6 py-2.5 rounded-xl text-sm transition-colors"
          >
            {loading ? 'Checking…' : '🛡️ Check Ingredients'}
          </button>
          {inci && (
            <button onClick={() => { setInci(''); setResult(null) }} className="text-sm text-gray-400 hover:text-gray-600">
              Clear
            </button>
          )}
          <p className="text-xs text-gray-400">
            Database: 200+ ingredients · Sources: EWG Skin Deep, EU Cosmetics Regulation, SCCS, IARC, DermNet NZ
          </p>
        </div>
      </div>

      {loading && <Spinner text="Cross-referencing against EWG/SCCS ingredient database…" />}

      {result && !loading && (
        <div className="space-y-5 fade-in">
          {/* Summary banner */}
          {result.summary && (
            <div className={`rounded-2xl p-5 flex flex-wrap gap-6 items-center ${OVERALL_RISK_STYLES[result.summary.overall_risk] || 'bg-gray-100'}`}>
              <div>
                <p className="text-sm font-medium opacity-80">Overall Risk</p>
                <p className="text-3xl font-bold font-display">{result.summary.overall_risk}</p>
              </div>
              <div className="grid grid-cols-4 gap-4 flex-1">
                {[
                  { label: 'Checked', val: result.summary.total_checked },
                  { label: 'Flagged', val: result.summary.total_flagged },
                  { label: '🔴 High', val: result.summary.high_risk },
                  { label: '🟡 Moderate', val: result.summary.moderate_risk },
                ].map(({ label, val }) => (
                  <div key={label} className="text-center">
                    <p className="text-2xl font-bold font-display">{val}</p>
                    <p className="text-xs opacity-80">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Flagged */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-display font-semibold text-red-700 mb-4">
                ⚠️ Flagged Ingredients ({result.flagged.length})
              </h3>
              {result.flagged.length === 0 ? (
                <p className="text-green-600 text-sm">No flagged ingredients found — clean product!</p>
              ) : (
                <div className="space-y-3">
                  {result.flagged
                    .sort((a,b) => {
                      const order = { HIGH: 0, MODERATE: 1, LOW: 2 }
                      return (order[a.risk_level] ?? 3) - (order[b.risk_level] ?? 3)
                    })
                    .map((f, i) => (
                      <EnhancedIngredientCard key={i} {...f} />
                    ))
                  }
                </div>
              )}
            </div>

            {/* Safe */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-display font-semibold text-green-700 mb-4">
                ✅ Safe Ingredients ({result.safe.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.safe.map((s, i) => <SafeBadge key={i} ingredient={s} />)}
              </div>
              {result.safe.length === 0 && (
                <p className="text-gray-400 text-sm">All ingredients were flagged for review.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Product catalogue */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-5">
          <div>
            <h2 className="font-display font-semibold text-gray-800">Irish Skincare Product Catalogue</h2>
            <p className="text-xs text-gray-400 mt-0.5">50 products from Boots IE, LookFantastic IE, McCabes Pharmacy. Click a row to view INCI list.</p>
          </div>
          <div className="flex gap-3 flex-wrap">
            <select value={catFilter} onChange={e => setCatFilter(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400">
              {CATEGORIES.map(c => <option key={c}>{c}</option>)}
            </select>
            <select value={stFilter} onChange={e => setStFilter(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400">
              {SKIN_TYPES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>

        {prodLoading ? <Spinner text="Loading products…" /> : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 text-xs text-gray-400 uppercase tracking-wider">
                  <th className="text-left py-3 px-3">Name</th>
                  <th className="text-left py-3 px-3">Brand</th>
                  <th className="text-left py-3 px-3">Category</th>
                  <th className="text-right py-3 px-3">Price</th>
                  <th className="text-center py-3 px-3">EWG</th>
                  <th className="text-left py-3 px-3">Skin Types</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => (
                  <>
                    <tr
                      key={p.id}
                      onClick={() => setExpanded(expanded === p.id ? null : p.id)}
                      className="border-b border-gray-50 hover:bg-teal-50 cursor-pointer transition-colors"
                    >
                      <td className="py-3 px-3 font-medium text-gray-800">{p.name}</td>
                      <td className="py-3 px-3 text-gray-500">{p.brand}</td>
                      <td className="py-3 px-3">
                        <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full text-xs">{p.category}</span>
                      </td>
                      <td className="py-3 px-3 text-right font-mono font-medium">€{p.price_eur.toFixed(2)}</td>
                      <td className={`py-3 px-3 text-center font-bold ${ewgColor(p.ewg_score)}`}>{p.ewg_score.toFixed(1)}</td>
                      <td className="py-3 px-3 text-gray-400 text-xs">{p.skin_type_tags.join(', ')}</td>
                    </tr>
                    {expanded === p.id && (
                      <tr key={`${p.id}-expanded`} className="bg-teal-50">
                        <td colSpan={6} className="px-4 py-3">
                          <div className="flex items-center justify-between mb-2">
                            <p className="text-xs font-medium text-gray-500">Full INCI List:</p>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                // pre-fill the ingredient checker above
                                const inciStr = p.inci_ingredients.join(', ')
                                window.scrollTo({ top: 0, behavior: 'smooth' })
                                // Use a custom event to pass to checker
                                document.getElementById('inci-textarea')?.focus()
                              }}
                              className="text-xs text-teal-600 hover:underline"
                            >
                              ↑ Check these ingredients
                            </button>
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {p.inci_ingredients.map((ing, i) => {
                              const flagged = ['alcohol denat.','parfum','fragrance','dmdm hydantoin','sodium lauryl sulfate','sls']
                                .includes(ing.toLowerCase()) || ing.toLowerCase().endsWith('paraben') || ing.toLowerCase().includes('isothiazolinone')
                              return (
                                <span key={i} className={`text-xs px-2 py-0.5 rounded-full border ${
                                  flagged ? 'bg-red-100 text-red-700 border-red-300' : 'bg-white text-gray-600 border-gray-200'
                                }`}>{ing}</span>
                              )
                            })}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-400 mt-3">Showing {filtered.length} of {products.length} products. Click a row to expand INCI list. 🔴 = flagged ingredient.</p>
          </div>
        )}
      </div>
    </div>
  )
}
