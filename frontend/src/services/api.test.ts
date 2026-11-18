import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, messageForStatus, NETWORK_ERROR_MESSAGE, predict } from './api'

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

  it('maps HTTP errors to Spanish messages and keeps the backend detail as secondary info', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'Invalid pixels' }),
    }))
    const error = await predict([]).catch((e: unknown) => e) as ApiError
    expect(error).toBeInstanceOf(ApiError)
    expect(error.message).toBe(messageForStatus(422))
    expect(error.message).toMatch(/24 × 40/)
    expect([error.status, error.detail]).toEqual([422, 'Invalid pixels'])
  })

  it('reads Pydantic validation details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: [{ msg: 'Field required' }, { msg: 'Input should be a valid list' }] }),
    }))
    await expect(predict([])).rejects.toMatchObject({ detail: 'Field required; Input should be a valid list' })
  })

  it.each([[413, /demasiado grande/], [415, /formato/], [503, /no está disponible/], [500, /problema/]])(
    'uses a clear message for status %i',
    (status, pattern) => expect(messageForStatus(status)).toMatch(pattern),
  )

  it('replaces network failures with a Spanish message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(predict([])).rejects.toMatchObject({ message: NETWORK_ERROR_MESSAGE })
  })
})
