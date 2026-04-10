import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import toast from 'react-hot-toast'
import { optimiseBasket } from '../api'
import { useApp } from '../context/AppContext'
import { ProductCard, Spinner } from '../components/index.jsx'
import { useNavigate } from 'react-router-dom'

const CLASS_NAMES = [
  'Melanoma','Melanocytic Nevi','Basal Cell Carcinoma','Actinic Keratosis',
  'Benign Keratosis','Dermatofibroma','Vascular Lesion','Squamous Cell Carcinoma','Unknown','General'
]
const CATEGORIES = ['cleanser','moisturiser','SPF','serum','toner']

// Condition-to-skincare guidance mapping
const CONDITION_SKINCARE_NOTES = {
  'Melanoma':               'Sun protection is critical. Mineral SPF50+ every day. Use gentle, fragrance-free formulations only. Avoid potential irritants that may mask lesion changes.',
  'Melanocytic Nevi':       'Standard skincare routine suitable. Daily SPF50+ recommended to prevent further UV-induced changes to existing nevi.',
  'Basal Cell Carcinoma':   'Mineral SPF50+ essential for prevention. Avoid harsh exfoliants and alcohol-based products near the lesion site.',
  'Actinic Keratosis':      'Daily mineral SPF50+ is mandatory. Ceramide-based moisturiser to support barrier function. Avoid AHAs, retinol, and fragrance on affected areas.',
  'Benign Keratosis':       'Gentle moisturising routine and SPF to prevent further sun damage. No aggressive exfoliants needed.',
  'Dermatofibroma':         'No specific skincare requirements. Standard gentle routine is suitable.',
  'Vascular Lesion':        'Rosacea-safe, alcohol-free, fragrance-free products. Mineral SPF50+. Avoid extreme temperatures and harsh actives.',
  'Squamous Cell Carcinoma':'Mineral SPF50+ is essential. Gentle, barrier-supporting cleanser and moisturiser. Avoid anything irritating near the affected site.',
  'Unknown':                'Use the most gentle formulations available. Mineral SPF, fragrance-free cleanser, and ceramide moisturiser are universally safe choices.',
  'General':                'Personalise your routine based on your skin type and concerns. Balance actives with barrier-supporting ingredients.',
}

// Recommended category priorities per condition
const CONDITION_RECOMMENDED_CATS = {
  'Melanoma':               ['cleanser','moisturiser','SPF'],
  'Melanocytic Nevi':       ['cleanser','moisturiser','SPF'],
  'Basal Cell Carcinoma':   ['cleanser','moisturiser','SPF'],
  'Actinic Keratosis':      ['cleanser','moisturiser','SPF'],
  'Benign Keratosis':       ['cleanser','moisturiser'],
  'Dermatofibroma':         ['cleanser','moisturiser'],
  'Vascular Lesion':        ['cleanser','moisturiser','SPF'],
  'Squamous Cell Carcinoma':['cleanser','moisturiser','SPF'],
  'Unknown':                ['cleanser','moisturiser'],
  'General':                ['cleanser','moisturiser'],
}

export default function Recommender() {
  const { detectedCondition, classifyResult } = useApp()
  const navigate = useNavigate()

  // Auto-populate from classifier if available
  const initialCondition = detectedCondition && detectedCondition !== '' ? detectedCondition : 'General'
  const initialCats = CONDITION_RECOMMENDED_CATS[initialCondition] || ['cleanser','moisturiser']

  const [condition, setCondition]   = useState(initialCondition)
  const [budget, setBudget]         = useState(80)
  const [categories, setCategories] = useState(initialCats)
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [autofilled, setAutofilled] = useState(!!detectedCondition && detectedCondition !== '')

  // Update if classifier result changes (e.g. user goes back and classifies again)
  useEffect(() => {
    if (detectedCondition && detectedCondition !== '') {
      setCondition(detectedCondition)
      setCategories(CONDITION_RECOMMENDED_CATS[detectedCondition] || ['cleanser','moisturiser'])
      setAutofilled(true)
      setResult(null) // clear old result when condition changes
    }
  }, [detectedCondition])

  const toggleCat = (cat) =>
    setCategories(prev => prev.includes(cat) ? prev.filter(c => c !== cat) : [...prev, cat])

  const handleOptimise = async () => {
    if (!categories.length) { toast.error('Select at least one product category.'); return }
    setLoading(true)
    try {
      const res = await optimiseBasket({ condition, budget, required_categories: categories })
      setResult(res.data)
      if (res.data.solve_status === 'Infeasible') {
        toast.error('No optimal solution — budget may be too low.')
      } else {
        toast.success('Optimal basket found!')
      }
    } catch (err) {
      toast.error('Optimisation failed. Is the backend running?')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const ewgChartData = result?.selected_products?.map(p => ({
    name: p.name.split(' ').slice(0,2).join(' '),
    ewg: p.ewg_score,
  })) || []

  const ewgColor = (score) => score <= 3 ? '#10b981' : score <= 6 ? '#f59e0b' : '#ef4444'

  const conditionNote = CONDITION_SKINCARE_NOTES[condition] || CONDITION_SKINCARE_NOTES['General']

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-display font-bold text-gray-900">Product Recommender</h1>
        <p className="text-gray-500 mt-1">LP-optimised skincare basket — maximum EWG safety score within your budget.</p>
      </div>

      {/* Auto-fill banner */}
      {autofilled && classifyResult && (
        <div className="mb-5 bg-teal-50 border border-teal-200 rounded-xl p-4 flex items-start gap-3">
          <span className="text-2xl mt-0.5">🔬</span>
          <div className="flex-1">
            <p className="text-teal-800 font-semibold text-sm">
              Auto-populated from your skin classifier result
            </p>
            <p className="text-teal-600 text-xs mt-1">
              Detected condition: <strong>{classifyResult.predicted_class}</strong> ({classifyResult.confidence}% confidence).
              Recommended product categories and condition have been pre-filled below. You can adjust them before optimising.
            </p>
          </div>
          <button
            onClick={() => navigate('/classifier')}
            className="text-xs text-teal-600 hover:text-teal-800 underline whitespace-nowrap"
          >
            Re-classify image
          </button>
        </div>
      )}

      {/* No classifier result — prompt user */}
      {!autofilled && (
        <div className="mb-5 bg-gray-50 border border-dashed border-gray-300 rounded-xl p-4 flex items-center gap-3">
          <span className="text-xl">💡</span>
          <p className="text-gray-600 text-sm flex-1">
            No skin condition detected yet. You can manually select a condition below, or{' '}
            <button onClick={() => navigate('/classifier')} className="text-teal-600 underline font-medium">
              go to the classifier
            </button>{' '}
            to auto-fill your condition from an image.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input panel */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-5 h-fit">
          <h2 className="font-display font-semibold text-gray-800">Configure</h2>

          <div>
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">Skin Condition</label>
            <select
              value={condition}
              onChange={e => { setCondition(e.target.value); setResult(null) }}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400"
            >
              {CLASS_NAMES.map(c => <option key={c}>{c}</option>)}
            </select>
            {autofilled && (
              <p className="text-xs text-teal-600 mt-1">✓ Auto-filled from classifier</p>
            )}
          </div>

          {/* Condition-specific skincare note */}
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-3">
            <p className="text-xs font-semibold text-blue-700 mb-1">💡 Skincare guidance for {condition}</p>
            <p className="text-xs text-blue-600 leading-relaxed">{conditionNote}</p>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">
              Budget: <span className="text-teal-600 font-bold">€{budget}</span>
            </label>
            <input
              type="range" min={10} max={200} step={5} value={budget}
              onChange={e => setBudget(+e.target.value)}
              className="w-full accent-teal-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1"><span>€10</span><span>€200</span></div>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-500 uppercase tracking-wide block mb-2">
              Required Categories
              {autofilled && <span className="ml-2 text-teal-500 normal-case">✓ condition-recommended</span>}
            </label>
            <div className="space-y-2">
              {CATEGORIES.map(cat => (
                <label key={cat} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={categories.includes(cat)}
                    onChange={() => toggleCat(cat)}
                    className="accent-teal-600"
                  />
                  <span className="text-sm capitalize text-gray-700">{cat}</span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleOptimise}
            disabled={loading}
            className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 text-white font-semibold py-3 rounded-xl text-sm transition-colors"
          >
            {loading ? 'Optimising…' : '⚡ Optimise Basket'}
          </button>

          <p className="text-xs text-gray-400 leading-relaxed">
            The LP solver (PuLP/CBC) selects the product combination with the highest aggregate EWG safety score that satisfies all category constraints within your budget.
          </p>
        </div>

        {/* Results panel */}
        <div className="lg:col-span-2 space-y-5">
          {loading && <Spinner text="Running LP optimisation (PuLP/CBC)…" />}

          {result && !loading && (
            <div className="space-y-5 fade-in">
              {/* Summary */}
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-5 flex flex-wrap gap-4 items-center justify-between">
                <div className="flex gap-6">
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide">Total Cost</p>
                    <p className="text-2xl font-display font-bold text-gray-900">€{result.total_price?.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide">Safety Score</p>
                    <p className="text-2xl font-display font-bold text-teal-600">{result.total_safety_score?.toFixed(1)}</p>
                    <p className="text-xs text-gray-400">aggregate EWG</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-400 uppercase tracking-wide">Products</p>
                    <p className="text-2xl font-display font-bold text-gray-700">{result.selected_products?.length || 0}</p>
                  </div>
                </div>
                <span className={`px-4 py-2 rounded-xl text-sm font-semibold ${
                  result.solve_status?.includes('Optimal')
                    ? 'bg-green-100 text-green-700'
                    : result.solve_status === 'Infeasible'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-amber-100 text-amber-700'
                }`}>
                  {result.solve_status?.includes('Optimal') ? '✅' : '⚠️'} {result.solve_status}
                </span>
              </div>

              {result.message && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-700">{result.message}</div>
              )}

              {/* Product cards */}
              {result.selected_products?.length > 0 && (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.selected_products.map(p => <ProductCard key={p.id} product={p} />)}
                  </div>

                  {/* EWG chart */}
                  <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-5">
                    <h3 className="font-display font-semibold text-gray-800 mb-1">EWG Safety Scores per Product</h3>
                    <p className="text-xs text-gray-400 mb-4">Lower EWG score = safer. Scores sourced from EWG Skin Deep database.</p>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={ewgChartData}>
                        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0,10]} tick={{ fontSize: 11 }} />
                        <Tooltip formatter={(v) => [`EWG ${v}`, 'Score']} />
                        <Bar dataKey="ewg" radius={[4,4,0,0]}>
                          {ewgChartData.map((entry, i) => (
                            <Cell key={i} fill={ewgColor(entry.ewg)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-gray-400 mt-2 text-center">🟢 EWG ≤3 (Safe) · 🟡 EWG 4–6 (Moderate) · 🔴 EWG &gt;6 (High concern)</p>
                  </div>

                  {/* Flagged in basket */}
                  {result.flagged_in_basket?.length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                      <p className="text-sm font-semibold text-red-700 mb-2">⚠️ Flagged ingredients in optimised basket:</p>
                      <div className="space-y-1">
                        {[...new Map(result.flagged_in_basket.map(f => [f.ingredient, f])).values()].map((f, i) => (
                          <div key={i} className="text-xs text-red-600">
                            <strong>{f.ingredient}</strong> ({f.risk_level}) — {f.explanation}
                          </div>
                        ))}
                      </div>
                      <p className="text-xs text-red-400 mt-2">Note: The LP optimiser minimises flagged ingredients, but budget constraints may require some compromise. Consider increasing budget for a cleaner basket.</p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {!result && !loading && (
            <div className="bg-white rounded-2xl border border-dashed border-gray-200 p-16 text-center text-gray-400">
              <div className="text-5xl mb-4">🛒</div>
              <p className="font-medium">Configure your preferences and click Optimise Basket</p>
              <p className="text-sm mt-1">The LP solver (PuLP/CBC) will find the safest products within your budget</p>
              {autofilled && (
                <p className="text-xs mt-3 text-teal-500">
                  ✓ Condition and categories pre-filled from classifier result
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
