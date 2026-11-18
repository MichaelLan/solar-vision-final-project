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
  version: string
  classes: { class: ClassName; label: string }[]
  input: { width: number; height: number; channels: number; normalization: string }
}

export interface ImageResult {
  id: string
  file: File
  previewUrl: string
  status: 'checking' | 'ready' | 'processing' | 'done' | 'error'
  result?: PredictionResponse
  error?: string
  /** True when the file failed client-side validation and will not be sent to the API. */
  rejected?: boolean
  /** Optional technical detail returned by the API, shown as secondary information. */
  errorDetail?: string
}

// `ink` is a darker tone of `color` used for text, so it meets WCAG AA on light backgrounds.
export const CLASS_DETAILS: Record<ClassName, { label: string; description: string; color: string; ink: string }> = {
  'No-Anomaly': {
    label: 'Sin anomalía',
    description: 'No se identifica un patrón térmico anómalo de las categorías estudiadas.',
    color: '#3d8976',
    ink: '#2f7262',
  },
  Cell: {
    label: 'Anomalía en una celda',
    description: 'Patrón térmico asociado con un punto caliente en una celda.',
    color: '#e29a51',
    ink: '#9a5c24',
  },
  'Cell-Multi': {
    label: 'Anomalía en varias celdas',
    description: 'Patrón térmico asociado con puntos calientes en varias celdas.',
    color: '#d67253',
    ink: '#b0533a',
  },
  Diode: {
    label: 'Patrón de diodo',
    description: 'Patrón térmico asociado con la activación de un diodo de bypass.',
    color: '#8e78b3',
    ink: '#6e5a96',
  },
}
