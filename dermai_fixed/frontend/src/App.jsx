import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Classifier from './pages/Classifier.jsx'
import IngredientGuard from './pages/IngredientGuard.jsx'
import Recommender from './pages/Recommender.jsx'
import Chatbot from './pages/Chatbot.jsx'
import Evaluation from './pages/Evaluation.jsx'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 pt-16">
        <Routes>
          <Route path="/" element={<Navigate to="/classifier" replace />} />
          <Route path="/classifier" element={<Classifier />} />
          <Route path="/ingredients" element={<IngredientGuard />} />
          <Route path="/recommender" element={<Recommender />} />
          <Route path="/chatbot" element={<Chatbot />} />
          <Route path="/evaluation" element={<Evaluation />} />
        </Routes>
      </main>
    </div>
  )
}
