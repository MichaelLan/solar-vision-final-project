export const CLASS_ORDER = ['No-Anomaly', 'Cell', 'Cell-Multi', 'Diode'] as const

export type ClassName = (typeof CLASS_ORDER)[number]

export interface Prediction {
  class: ClassName
  label: string
  confidence: number
}

export interface PredictionResponse {
  prediction: Prediction
  probabilities: Record<ClassName, number>
  model_version: string
}

export interface ModelInfo {
  version?: string
  model_version?: string
  classes?: string[]
  input_shape?: number[]
}

export interface ImageResult {
  id: string
  file: File
  previewUrl: string
  status: 'ready' | 'processing' | 'done' | 'error'
  result?: PredictionResponse
  error?: string
}

export const CLASS_DETAILS: Record<ClassName, { label: string; description: string; color: string }> = {
  'No-Anomaly': {
    label: 'Sin anomalía',
    description: 'No se identifica un patrón térmico anómalo de las categorías estudiadas.',
    color: '#3d8976',
  },
  Cell: {
    label: 'Anomalía en una celda',
    description: 'Patrón térmico asociado con un punto caliente en una celda.',
    color: '#e29a51',
  },
  'Cell-Multi': {
    label: 'Anomalía en varias celdas',
    description: 'Patrón térmico asociado con puntos calientes en varias celdas.',
    color: '#d67253',
  },
  Diode: {
    label: 'Patrón de diodo',
    description: 'Patrón térmico asociado con la activación de un diodo de bypass.',
    color: '#8e78b3',
  },
}
