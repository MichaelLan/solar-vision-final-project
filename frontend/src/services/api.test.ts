import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, predict } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('prediction API', () => {
  it('sends image dimensions and pixels to the real endpoint contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ prediction: { class: 'Cell', confidence: 0.8, label: 'Cell' }, probabilities: {}, model_version: '1.0.0' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    await predict([0, 1])
    expect(fetchMock).toHaveBeenCalledWith('/api/predict', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ pixels: [0, 1], width: 24, height: 40 }),
    }))
  })

  it('propagates HTTP errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'Invalid pixels' }),
    }))
    await expect(predict([])).rejects.toEqual(new ApiError('Invalid pixels', 422))
  })
})
