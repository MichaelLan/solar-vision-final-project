import { describe, expect, it } from 'vitest'
import { dimensionError, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_COUNT, rgbaToPixels, validateImage } from './image'

describe('image input', () => {
  it('uses the actual dataset image orientation', () => {
    expect([IMAGE_WIDTH, IMAGE_HEIGHT, PIXEL_COUNT]).toEqual([24, 40, 960])
  })
  it('rejects unsupported and empty files', () => {
    expect(validateImage(new File(['x'], 'x.txt', { type: 'text/plain' }))).toMatch(/PNG/)
    expect(validateImage(new File([], 'x.png', { type: 'image/png' }))).toMatch(/vacía/)
  })

  it('explains wrong pixel dimensions in Spanish', () => {
    expect(dimensionError(24, 40)).toBeNull()
    expect(dimensionError(40, 24)).toMatch(/mide 40 × 24 px y debe medir 24 × 40 px/)
  })

  it('converts RGB to grayscale in row-major order', () => {
    const rgba = new Uint8ClampedArray(PIXEL_COUNT * 4)
    rgba.set([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255])
    const pixels = rgbaToPixels(rgba)
    expect(pixels).toHaveLength(PIXEL_COUNT)
    expect(pixels.slice(0, 3)).toEqual([76 / 255, 150 / 255, 29 / 255])
    expect(pixels[3]).toBe(0)
  })

  it('rejects a tensor with the wrong dimensions', () => {
    expect(() => rgbaToPixels(new Uint8ClampedArray(8))).toThrow(/Dimensiones/)
  })
})
