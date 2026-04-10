import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import toast from 'react-hot-toast'
import { classifyImage } from '../api'
import { useApp } from '../context/AppContext'
import { Disclaimer, Spinner, StatCard } from '../components/index.jsx'
import { useNavigate } from 'react-router-dom'

const CLASS_NAMES = [
  'Melanoma','Melanocytic Nevi','Basal Cell Carcinoma','Actinic Keratosis',
  'Benign Keratosis','Dermatofibroma','Vascular Lesion','Squamous Cell Carcinoma','Unknown'
]

// Condition info cards for clinical context
const CONDITION_INFO = {
  'Melanoma':              { color:'red',   icon:'🔴', desc:'Most serious skin cancer; requires urgent dermatologist review. SPF50+ daily is essential.', care:'Gentle, fragrance-free skincare only. Avoid harsh exfoliants.' },
  'Melanocytic Nevi':      { color:'green', icon:'🟢', desc:'Common mole. Usually benign but monitor for ABCDEs (asymmetry, border, colour, diameter, evolution).', care:'Standard skincare routine; daily SPF recommended.' },
  'Basal Cell Carcinoma':  { color:'orange',icon:'🟠', desc:'Most common skin cancer; rarely spreads but requires treatment. Seek dermatologist evaluation.', care:'Mineral SPF50+; gentle non-irritating cleanser and moisturiser.' },
  'Actinic Keratosis':     { color:'orange',icon:'🟠', desc:'Pre-cancerous lesion caused by UV damage; treatable. Dermatologist review recommended.', care:'Mineral SPF50+, ceramide moisturiser; avoid AHAs/retinol on affected area.' },
  'Benign Keratosis':      { color:'green', icon:'🟢', desc:'Non-cancerous skin growth (seborrhoeic keratosis). No treatment needed unless symptomatic.', care:'Gentle moisturising routine; SPF to prevent further sun damage.' },
  'Dermatofibroma':        { color:'green', icon:'🟢', desc:'Benign fibrous nodule; usually harmless. Removal possible if symptomatic.', care:'No specific skincare required; standard routine suitable.' },
  'Vascular Lesion':       { color:'blue',  icon:'🔵', desc:'Blood vessel-related skin condition. Usually benign; laser treatment available if desired.', care:'Gentle, rosacea-safe products; avoid alcohol and fragrance.' },
  'Squamous Cell Carcinoma':{ color:'red',  icon:'🔴', desc:'Second most common skin cancer; requires treatment. Prompt dermatologist referral recommended.', care:'Mineral SPF50+; gentle barrier-supporting routine only.' },
  'Unknown':               { color:'gray',  icon:'⚪', desc:'Unable to classify with confidence. Please consult a dermatologist for professional evaluation.', care:'Maintain a gentle, fragrance-free routine. Use mineral SPF.' },
}

const CONFIDENCE_THRESHOLDS = {
  high: 70,
  medium: 40,
}

function ConfidenceBadge({ confidence }) {
  if (confidence >= CONFIDENCE_THRESHOLDS.high) {
    return <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium">High Confidence</span>
  } else if (confidence >= CONFIDENCE_THRESHOLDS.medium) {
    return <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-700 font-medium">Medium Confidence</span>
  }
  return <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700 font-medium">Low Confidence</span>
}

export default function Classifier() {
  const { setDetectedCondition, setClassifyResult, setDemoMode } = useApp()
  const navigate = useNavigate()
  const [file, setFile]         = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return
    const f = accepted[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': [] }, multiple: false
  })

  const handleAnalyse = async () => {
    if (!file) { toast.error('Please upload an image first.'); return }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await classifyImage(fd)
      const data = res.data

      if (data.error) {
        setError(data.error + (data.detail ? ': ' + data.detail : ''))
        toast.error('Classification failed.')
        return
      }

      setResult(data)
      setClassifyResult(data)
      setDetectedCondition(data.predicted_class)
      setDemoMode(data.demo_mode)
      toast.success(`Classified as: ${data.predicted_class}`)
    } catch (err) {
      const msg = err.response
        ? `Backend error: ${err.response.status} — ${JSON.stringify(err.response.data)}`
        : `Network error — is the backend running on localhost:8000?`
      setError(msg)
      toast.error('Classification failed.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const chartData = result
    ? Object.entries(result.all_class_probs || {})
        .map(([name, prob]) => ({ name: name.split(' ').slice(0,2).join(' '), prob: +(prob * 100).toFixed(1), full: name }))
        .sort((a, b) => b.prob - a.prob)
    : []

  const conditionInfo = result ? (CONDITION_INFO[result.predicted_class] || CONDITION_INFO['Unknown']) : null

  const colorMap = { red:'bg-red-600', orange:'bg-orange-500', green:'bg-green-600', blue:'bg-blue-600', gray:'bg-gray-500' }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-display font-bold text-gray-900">Skin Condition Classifier</h1>
        <p className="text-gray-500 mt-1">Upload a dermoscopy image to classify using our deep ensemble model (EfficientNet-B7 + ViT-B/16).</p>
      </div>

      <Disclaimer />

      {result?.demo_mode && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-amber-700 text-sm">
          ⚙️ <strong>Demo mode</strong> — models are using ImageNet pretrained weights, not fine-tuned on ISIC data.
          Use the Evaluation page to download ISIC training data and retrain for clinical accuracy.
        </div>
      )}

      {/* Upload */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-teal-500 bg-teal-50' : 'border-gray-300 hover:border-teal-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          {preview ? (
            <img src={preview} alt="preview" className="max-h-64 mx-auto rounded-lg object-contain" />
          ) : (
            <div className="space-y-3">
              <div className="text-5xl">🔬</div>
              <p className="text-gray-500 font-medium">
                {isDragActive ? 'Drop the image here…' : 'Drag & drop a skin image or click to browse'}
              </p>
              <p className="text-xs text-gray-400">PNG, JPG, JPEG supported</p>
            </div>
          )}
        </div>
        <button
          onClick={handleAnalyse}
          disabled={!file || loading}
          className="mt-4 w-full bg-teal-600 hover:bg-teal-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-colors text-sm"
        >
          {loading ? 'Analysing…' : '🔍 Analyse Image'}
        </button>
      </div>

      {/* Error state */}
      {error && !loading && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
          <strong>⚠️ Error:</strong> {error}
          <p className="mt-1 text-xs text-red-500">Make sure the backend is running: <code>uvicorn main:app --reload --port 8000</code></p>
        </div>
      )}

      {/* Results */}
      {loading && <Spinner text="Running ensemble classifier (EfficientNet-B7 + ViT-B/16)…" />}

      {result && !loading && (
        <div className="space-y-6 fade-in">
          {/* Condition Info Banner */}
          {conditionInfo && (
            <div className={`rounded-2xl p-5 text-white ${colorMap[conditionInfo.color] || 'bg-gray-600'}`}>
              <div className="flex items-start gap-4">
                <span className="text-4xl">{conditionInfo.icon}</span>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2 flex-wrap">
                    <h2 className="text-xl font-bold font-display">{result.predicted_class}</h2>
                    <ConfidenceBadge confidence={result.confidence} />
                    <span className="text-white/80 text-sm font-mono">{result.confidence}% confidence</span>
                  </div>
                  <p className="text-white/90 text-sm leading-relaxed mb-2">{conditionInfo.desc}</p>
                  <p className="text-white/80 text-xs"><strong>Skincare guidance:</strong> {conditionInfo.care}</p>
                </div>
              </div>
              {result.confidence < CONFIDENCE_THRESHOLDS.medium && (
                <div className="mt-3 bg-white/20 rounded-lg p-3 text-sm">
                  ⚠️ <strong>Low confidence result.</strong> The model is uncertain. This image may be outside the training distribution or show mixed features. Always consult a dermatologist.
                </div>
              )}
            </div>
          )}

          {/* CTA → Recommender */}
          <div className="bg-teal-50 border border-teal-200 rounded-xl p-4 flex items-center justify-between gap-4">
            <div>
              <p className="text-teal-800 font-semibold text-sm">💊 Get personalised skincare recommendations</p>
              <p className="text-teal-600 text-xs mt-0.5">Your detected condition ({result.predicted_class}) has been saved. The Recommender will auto-fill it.</p>
            </div>
            <button
              onClick={() => navigate('/recommender')}
              className="bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold px-4 py-2 rounded-xl whitespace-nowrap transition-colors"
            >
              Go to Recommender →
            </button>
          </div>

          {/* Two column layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Images */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
              <h2 className="font-display font-semibold text-gray-800">Image Analysis</h2>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-gray-400 mb-2 text-center">Original</p>
                  <img src={preview} alt="original" className="w-full rounded-lg object-cover aspect-square" />
                </div>
                <div>
                  <p className="text-xs text-gray-400 mb-2 text-center">Grad-CAM Heatmap</p>
                  {result.gradcam_image ? (
                    <img
                      src={`data:image/png;base64,${result.gradcam_image}`}
                      alt="gradcam"
                      className="w-full rounded-lg object-cover aspect-square"
                    />
                  ) : (
                    <div className="w-full aspect-square rounded-lg bg-gradient-to-br from-green-300 via-yellow-300 to-red-400 flex items-center justify-center text-white text-xs font-medium">
                      Grad-CAM Overlay
                    </div>
                  )}
                </div>
              </div>
              <p className="text-xs text-gray-400">Grad-CAM highlights the image regions most influential to the classification decision, enabling interpretable AI output.</p>
            </div>

            {/* Prediction */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 space-y-4">
              <h2 className="font-display font-semibold text-gray-800">Class Probability Distribution</h2>
              <p className="text-xs text-gray-400">All 9 ISIC diagnostic classes — sorted by probability</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 20 }}>
                  <XAxis type="number" domain={[0,100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={110} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="prob" radius={[0,4,4,0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={entry.full === result.predicted_class ? '#0d9488' : '#99f6e4'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <StatCard label="Deep Ensemble (EfficientNet-B4 + ViT-B/16)" value={result.ensemble_label} color="blue" />
            <StatCard label="Traditional ML (SVM + RF HOG/LBP)" value={result.traditional_ml_label} color="purple" />
            <StatCard
              label="Anomaly Score"
              value={result.anomaly_score}
              sub={result.anomaly_score >= 0.05 ? '⚠️ Potentially anomalous — out-of-distribution' : '✓ Normal range — within training distribution'}
              color={result.anomaly_score >= 0.05 ? 'red' : 'green'}
            />
          </div>

          {/* Confidence reliability note */}
          {result.confidence_reliability && (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-xs text-gray-600">
              <span className="font-semibold text-gray-700">🎯 Confidence reliability: </span>
              {result.confidence_reliability}
              {result.confidence_raw !== result.confidence && (
                <span className="ml-2 text-gray-400">(raw: {result.confidence_raw}% → calibrated: {result.confidence}%)</span>
              )}
            </div>
          )}

          {/* Model agreement note */}
          {result.ensemble_label !== result.traditional_ml_label && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-amber-700 text-sm">
              ⚠️ <strong>Model disagreement:</strong> Deep ensemble ({result.ensemble_label}) and traditional ML ({result.traditional_ml_label}) disagree. This is common in ambiguous cases and highlights the value of ensemble methods. The deep ensemble prediction is used as the final output.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
