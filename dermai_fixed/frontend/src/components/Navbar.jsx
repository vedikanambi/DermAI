import { NavLink } from 'react-router-dom'
import { useApp } from '../context/AppContext'

const LINKS = [
  { to: '/classifier',   label: '🔬 Classifier',      title: 'Skin Condition Classifier' },
  { to: '/ingredients',  label: '🛡️ Ingredient Guard', title: 'EWG Ingredient Safety Checker' },
  { to: '/recommender',  label: '🛒 Recommender',      title: 'LP-Optimised Product Recommender' },
  { to: '/chatbot',      label: '💬 AI Assistant',     title: 'RAG Chatbot' },
  { to: '/evaluation',   label: '📊 Evaluation',       title: 'Model Evaluation Dashboard' },
]

export default function Navbar() {
  const { detectedCondition, demoMode } = useApp()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 h-16 flex items-center px-4 gap-4">
      {/* Logo */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-2xl">🩺</span>
        <div>
          <span className="font-display font-bold text-gray-900 text-lg">DermAI</span>
          <span className="font-display font-bold text-teal-600 text-lg"> Guard</span>
          <span className="ml-2 text-xs bg-teal-100 text-teal-700 px-2 py-0.5 rounded-full font-medium hidden sm:inline">Research Prototype</span>
        </div>
      </div>

      {/* Nav links */}
      <div className="flex items-center gap-1 flex-1 overflow-x-auto">
        {LINKS.map(({ to, label, title }) => (
          <NavLink
            key={to}
            to={to}
            title={title}
            className={({ isActive }) =>
              `whitespace-nowrap px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-teal-50 text-teal-700 border border-teal-200'
                  : 'text-gray-500 hover:text-gray-800 hover:bg-gray-50'
              }`
            }
          >
            {label}
          </NavLink>
        ))}
      </div>

      {/* Right: detected condition pill + demo badge */}
      <div className="flex items-center gap-2 shrink-0">
        {detectedCondition && (
          <NavLink to="/recommender" title="Go to Recommender with this condition"
            className="hidden md:flex items-center gap-1 bg-teal-50 border border-teal-200 text-teal-700 px-3 py-1 rounded-full text-xs font-medium hover:bg-teal-100 transition-colors">
            🔬 <span className="max-w-28 truncate">{detectedCondition}</span>
          </NavLink>
        )}
        {demoMode && (
          <span className="hidden sm:inline text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-full font-medium">
            Demo Mode
          </span>
        )}
      </div>
    </nav>
  )
}
