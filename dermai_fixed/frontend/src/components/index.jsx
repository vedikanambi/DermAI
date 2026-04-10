// Disclaimer
export function Disclaimer() {
  return (
    <div className="border border-red-300 bg-red-50 rounded-xl p-4 flex gap-3 items-start">
      <span className="text-red-500 text-xl mt-0.5">⚠️</span>
      <p className="text-red-700 text-sm leading-relaxed">
        <strong>Research Prototype Only.</strong> This tool is for educational and research purposes only.
        It is <strong>NOT</strong> a medical diagnostic device and must not be used for clinical decisions.
        Always consult a qualified dermatologist before making any skincare or medical decisions.
      </p>
    </div>
  )
}

// IngredientBadge (legacy, used in some places)
export function IngredientBadge({ ingredient, risk_level, explanation }) {
  const RISK_STYLES = {
    HIGH:     'bg-red-100 text-red-800 border-red-300',
    MODERATE: 'bg-amber-100 text-amber-800 border-amber-300',
    LOW:      'bg-yellow-50 text-yellow-700 border-yellow-300',
  }
  return (
    <div className={`border rounded-lg p-3 text-sm ${RISK_STYLES[risk_level] || 'bg-gray-100 text-gray-700 border-gray-300'}`}>
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="font-semibold">{ingredient}</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-bold ${
          risk_level === 'HIGH' ? 'bg-red-500 text-white' :
          risk_level === 'MODERATE' ? 'bg-amber-500 text-white' :
          'bg-yellow-400 text-yellow-900'
        }`}>{risk_level}</span>
      </div>
      {explanation && <p className="text-xs opacity-80 leading-relaxed">{explanation}</p>}
    </div>
  )
}

// SafeBadge
export function SafeBadge({ ingredient }) {
  return (
    <span className="inline-block bg-green-50 text-green-700 border border-green-200 rounded-full px-3 py-1 text-xs font-medium">
      ✓ {ingredient}
    </span>
  )
}

// Expanded flagging for ProductCard — covers all new ingredient patterns
const isFlagged = (ing) => {
  const l = ing.toLowerCase().trim()
  return (
    ['alcohol denat.', 'denatured alcohol', 'sd alcohol', 'parfum', 'fragrance',
     'dmdm hydantoin', 'imidazolidinyl urea', 'diazolidinyl urea', 'quaternium-15',
     'sodium lauryl sulfate', 'sls', 'triclosan', 'triclocarban',
     'methylisothiazolinone', 'methylchloroisothiazolinone', 'formaldehyde',
     'benzophenone-3', 'oxybenzone', 'butylphenyl methylpropional', 'lilial',
     'coal tar', 'lead acetate', 'dibutyl phthalate'].includes(l) ||
    l.endsWith('paraben') ||
    l.includes('isothiazolinone')
  )
}

// ProductCard
export function ProductCard({ product, showFlag = true }) {
  const ewgColor =
    product.ewg_score <= 3 ? 'bg-green-100 text-green-700 border-green-300' :
    product.ewg_score <= 6 ? 'bg-amber-100 text-amber-700 border-amber-300' :
                              'bg-red-100 text-red-700 border-red-300'

  const categoryColors = {
    cleanser:    'bg-blue-100 text-blue-700',
    moisturiser: 'bg-purple-100 text-purple-700',
    SPF:         'bg-orange-100 text-orange-700',
    serum:       'bg-pink-100 text-pink-700',
    toner:       'bg-teal-100 text-teal-700',
  }

  const flaggedIngs = product.inci_ingredients?.filter(isFlagged) || []

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-2 mb-3">
        <div>
          <h3 className="font-semibold text-gray-900 text-sm leading-tight">{product.name}</h3>
          <p className="text-gray-400 text-xs mt-0.5">{product.brand}</p>
        </div>
        <div className="text-right shrink-0">
          <p className="text-teal-700 font-bold text-lg">€{product.price_eur.toFixed(2)}</p>
          <a href={product.url} target="_blank" rel="noopener noreferrer"
            className="text-xs text-teal-500 hover:underline">View →</a>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        <span className={`text-xs px-2 py-1 rounded-full font-medium ${categoryColors[product.category] || 'bg-gray-100 text-gray-600'}`}>
          {product.category}
        </span>
        <span className={`text-xs px-2 py-1 rounded-full font-medium border ${ewgColor}`}>
          EWG {product.ewg_score.toFixed(1)}
        </span>
      </div>

      <div className="flex flex-wrap gap-1 mb-3">
        {product.skin_type_tags.map(t => (
          <span key={t} className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">{t}</span>
        ))}
      </div>

      {showFlag && (
        flaggedIngs.length > 0 ? (
          <p className="text-xs text-red-600 mt-2">⚠️ Contains flagged: {flaggedIngs.slice(0,3).join(', ')}{flaggedIngs.length>3?` +${flaggedIngs.length-3} more`:''}</p>
        ) : (
          <p className="text-xs text-green-600 mt-2">✓ No flagged ingredients detected</p>
        )
      )}
    </div>
  )
}

// ChatBubble
export function ChatBubble({ role, content, chunks }) {
  const isUser = role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'} mb-4`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0 mt-1">AI</div>
      )}
      <div className={`max-w-[75%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-teal-600 text-white rounded-tr-sm'
            : 'bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm'
        }`}>
          {content}
        </div>
        {!isUser && chunks && chunks.length > 0 && (
          <details className="text-xs mt-1">
            <summary className="cursor-pointer text-gray-400 hover:text-teal-600 transition-colors">
              📚 Retrieved sources ({chunks.length})
            </summary>
            <div className="mt-2 space-y-2">
              {chunks.map((chunk, i) => (
                <blockquote key={i} className="border-l-2 border-teal-300 pl-3 text-gray-500 italic text-xs leading-relaxed">
                  <strong className="not-italic text-teal-600">Source {i + 1}:</strong> {typeof chunk === 'string' ? chunk.slice(0, 200) : JSON.stringify(chunk).slice(0,200)}…
                </blockquote>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

// TypingIndicator
export function TypingIndicator() {
  return (
    <div className="flex gap-3 mb-4">
      <div className="w-8 h-8 rounded-full bg-teal-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">AI</div>
      <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-center h-4">
          {[0,1,2].map(i => (
            <div key={i} className="w-2 h-2 rounded-full bg-teal-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      </div>
    </div>
  )
}

// Spinner
export function Spinner({ text = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center gap-3 py-12">
      <div className="w-10 h-10 border-4 border-teal-200 border-t-teal-600 rounded-full animate-spin" />
      <p className="text-sm text-gray-500 font-medium">{text}</p>
    </div>
  )
}

// StatCard
export function StatCard({ label, value, sub, color = 'teal' }) {
  const colors = {
    teal:   'bg-teal-50 border-teal-200 text-teal-700',
    blue:   'bg-blue-50 border-blue-200 text-blue-700',
    purple: 'bg-purple-50 border-purple-200 text-purple-700',
    green:  'bg-green-50 border-green-200 text-green-700',
    red:    'bg-red-50 border-red-200 text-red-700',
    amber:  'bg-amber-50 border-amber-200 text-amber-700',
  }
  return (
    <div className={`border rounded-xl p-4 ${colors[color] || colors.teal}`}>
      <p className="text-xs font-medium opacity-70 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold font-display">{value}</p>
      {sub && <p className="text-xs mt-1 opacity-60">{sub}</p>}
    </div>
  )
}
