import type { ModelInfo, PredictionResponse } from '../types/model'
import { IMAGE_HEIGHT, IMAGE_WIDTH } from './image'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

const STATUS_MESSAGES: Record<number, string> = {
  400: 'La solicitud no es válida. Vuelve a cargar la imagen e inténtalo de nuevo.',
  413: 'La imagen enviada es demasiado grande para el servicio.',
  415: 'El servicio no admite el formato enviado.',
  422: 'El servicio rechazó la imagen: debe ser de 24 × 40 px con valores de píxel válidos.',
  429: 'Demasiadas solicitudes seguidas. Espera unos segundos e inténtalo de nuevo.',
  503: 'El modelo no está disponible en este momento. Inténtalo más tarde.',
}

export const NETWORK_ERROR_MESSAGE = 'No se pudo conectar con el servicio. Comprueba tu conexión e inténtalo de nuevo.'

/** Spanish, user-facing message for an HTTP status; backend details stay in `ApiError.detail`. */
export function messageForStatus(status: number): string {
  if (STATUS_MESSAGES[status]) return STATUS_MESSAGES[status]
  if (status >= 500) return 'El servicio tuvo un problema al procesar la imagen. Inténtalo más tarde.'
  return 'No se pudo completar la solicitud.'
}

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number, public readonly detail?: string) {
    super(message)
    this.name = 'ApiError'
  }
}

function extractDetail(body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((entry) => (entry && typeof entry === 'object' && 'msg' in entry ? String(entry.msg) : ''))
      .filter(Boolean)
      .join('; ')
  }
  return ''
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = ''
    try {
      detail = extractDetail(await response.json())
    } catch {}
    throw new ApiError(messageForStatus(response.status), response.status, detail || undefined)
  }
  return response.json() as Promise<T>
}

async function request(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError(NETWORK_ERROR_MESSAGE)
  }
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const response = await request(`${API_BASE}/api/model/info`, { signal })
  return parseResponse<ModelInfo>(response)
}

export async function predict(pixels: number[], signal?: AbortSignal): Promise<PredictionResponse> {
  const response = await request(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pixels, width: IMAGE_WIDTH, height: IMAGE_HEIGHT }),
    signal,
  })
  return parseResponse<PredictionResponse>(response)
}
