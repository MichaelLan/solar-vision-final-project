import type { ModelInfo, PredictionResponse } from '../types/model'
import { IMAGE_HEIGHT, IMAGE_WIDTH } from './image'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json() as { detail?: string | { msg?: string }[] }
      detail = typeof body.detail === 'string' ? body.detail : ''
    } catch {}
    throw new ApiError(detail || 'No se pudo completar la solicitud.', response.status)
  }
  return response.json() as Promise<T>
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE}/api/model/info`, { signal })
  return parseResponse<ModelInfo>(response)
}

export async function predict(pixels: number[], signal?: AbortSignal): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pixels, width: IMAGE_WIDTH, height: IMAGE_HEIGHT }),
    signal,
  })
  return parseResponse<PredictionResponse>(response)
}
