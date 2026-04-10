import { useState, useRef, useEffect } from 'react'
import { sendChat } from '../api'
import { useApp } from '../context/AppContext'
import { ChatBubble, TypingIndicator } from '../components/index.jsx'

const STARTERS = [
  "What ingredients should I avoid for eczema?",
  "Is my product safe for sensitive skin?",
  "What routine suits acne-prone skin?",
  "What does Niacinamide do?",
]

export default function Chatbot() {
  const { detectedCondition, chatHistory, setChatHistory } = useApp()
  const [input, setInput]     = useState('')
  const [typing, setTyping]   = useState(false)
  const bottomRef             = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, typing])

  const handleSend = async (msg) => {
    const text = (msg || input).trim()
    if (!text) return
    setInput('')

    const userMsg = { role: 'user', content: text }
    const newHistory = [...chatHistory, userMsg]
    setChatHistory(newHistory)
    setTyping(true)

    try {
      const res = await sendChat({
        user_message: text,
        detected_condition: detectedCondition || 'Not detected',
        conversation_history: newHistory.map(m => ({ role: m.role, content: m.content })),
      })
      const assistantMsg = {
        role: 'assistant',
        content: res.data.assistant_message,
        chunks: res.data.retrieved_chunks,
      }
      setChatHistory([...newHistory, assistantMsg])
    } catch (err) {
      setChatHistory([...newHistory, {
        role: 'assistant',
        content: '⚠️ Could not reach the AI service. Please ensure the backend is running and Azure OpenAI is configured.',
        chunks: [],
      }])
    } finally {
      setTyping(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="flex h-[calc(100vh-64px)]">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col p-5 gap-5 flex-shrink-0">
        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Detected Condition</p>
          {detectedCondition ? (
            <span className="inline-block bg-teal-100 text-teal-700 px-3 py-1.5 rounded-lg text-sm font-semibold">{detectedCondition}</span>
          ) : (
            <span className="text-gray-400 text-sm italic">Run classifier first</span>
          )}
        </div>

        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Model</p>
          <p className="text-xs text-gray-600">GPT-4o via Azure OpenAI</p>
          <p className="text-xs text-gray-400 mt-1">RAG-augmented with ChromaDB</p>
        </div>

        <div>
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Quick Prompts</p>
          <div className="space-y-2">
            {STARTERS.map(s => (
              <button
                key={s}
                onClick={() => handleSend(s)}
                className="w-full text-left text-xs text-gray-600 bg-gray-50 hover:bg-teal-50 hover:text-teal-700 border border-gray-200 hover:border-teal-300 rounded-lg px-3 py-2 transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={() => setChatHistory([])}
          className="mt-auto border border-red-200 text-red-500 hover:bg-red-50 rounded-lg py-2 text-sm transition-colors"
        >
          Clear Chat
        </button>
      </aside>

      {/* Chat area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-2 scrollbar-thin">
          {chatHistory.length === 0 && !typing && (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 gap-4">
              <div className="text-6xl">💬</div>
              <div>
                <p className="font-display font-semibold text-gray-600 text-lg">Ask me anything about skincare</p>
                <p className="text-sm mt-1">I'll use your detected condition and ingredient knowledge to help.</p>
              </div>
            </div>
          )}
          {chatHistory.map((msg, i) => (
            <ChatBubble key={i} role={msg.role} content={msg.content} chunks={msg.chunks} />
          ))}
          {typing && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="border-t border-gray-200 bg-white p-4">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={typing}
              placeholder="Ask about ingredients, routines, or your skin condition… (Enter to send)"
              rows={1}
              className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-teal-500 disabled:bg-gray-50"
            />
            <button
              onClick={() => handleSend()}
              disabled={typing || !input.trim()}
              className="bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-5 py-3 rounded-xl font-semibold text-sm transition-colors"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
