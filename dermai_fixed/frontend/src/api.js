import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8000' })

export const classifyImage = (formData) =>
  api.post('/classify', formData, { headers: { 'Content-Type': 'multipart/form-data' } })

export const flagIngredients = (inciList) =>
  api.post('/flag-ingredients', { inci_list: inciList })

export const getProducts = (params = {}) =>
  api.get('/products', { params })

export const optimiseBasket = (payload) =>
  api.post('/optimise', payload)

export const sendChat = (payload) =>
  api.post('/chat', payload)

export const getEvaluation = () =>
  api.get('/evaluation')

export const getDatasetStats = () =>
  api.get('/dataset/stats')

export const getDemoImages = (diagnosis) =>
  api.get(`/dataset/demo-images/${encodeURIComponent(diagnosis)}`)

export const startDownload = (limit_per_class = 200) =>
  api.post('/dataset/download', { limit_per_class })

export const getDownloadStatus = () =>
  api.get('/dataset/download-status')

export const checkHealth = () =>
  api.get('/health')

export default api
