import { createContext, useContext, useState } from 'react'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [detectedCondition, setDetectedCondition] = useState('')
  const [classifyResult, setClassifyResult] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [optimiserResult, setOptimiserResult] = useState(null)
  const [demoMode, setDemoMode] = useState(true)

  return (
    <AppContext.Provider value={{
      detectedCondition, setDetectedCondition,
      classifyResult, setClassifyResult,
      chatHistory, setChatHistory,
      optimiserResult, setOptimiserResult,
      demoMode, setDemoMode,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
