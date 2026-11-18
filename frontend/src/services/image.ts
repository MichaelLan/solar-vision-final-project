export const IMAGE_WIDTH = 24
export const IMAGE_HEIGHT = 40
export const PIXEL_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT
export const MAX_FILE_SIZE = 5 * 1024 * 1024
export const MAX_BATCH_SIZE = 20

const ALLOWED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp'])

export function rgbaToPixels(rgba: Uint8ClampedArray): number[] {
  if (rgba.length !== PIXEL_COUNT * 4) throw new Error('Dimensiones de imagen incorrectas.')
  const pixels = new Array<number>(PIXEL_COUNT)
  for (let i = 0; i < PIXEL_COUNT; i += 1) {
    const offset = i * 4
    pixels[i] = Math.round(
      0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2],
    ) / 255
  }
  return pixels
}

export function validateImage(file: File): string | null {
  if (!ALLOWED_TYPES.has(file.type)) return 'Usa una imagen PNG, JPEG o WebP.'
  if (file.size > MAX_FILE_SIZE) return 'La imagen supera el límite de 5 MB.'
  if (file.size === 0) return 'La imagen está vacía.'
  return null
}

export function dimensionError(width: number, height: number): string | null {
  if (width === IMAGE_WIDTH && height === IMAGE_HEIGHT) return null
  return `Tamaño no admitido: mide ${width} × ${height} px y debe medir ${IMAGE_WIDTH} × ${IMAGE_HEIGHT} px, como las imágenes de entrenamiento.`
}

async function decodeImage(url: string): Promise<HTMLImageElement> {
  const image = new Image()
  image.src = url
  try {
    await image.decode()
  } catch {
    throw new Error('No se pudo leer la imagen; el archivo puede estar dañado.')
  }
  return image
}

/** Full client-side check (type, size and pixel dimensions) used when files are added. */
export async function checkImage(file: File): Promise<string | null> {
  const basicError = validateImage(file)
  if (basicError) return basicError
  const url = URL.createObjectURL(file)
  try {
    const image = await decodeImage(url)
    return dimensionError(image.naturalWidth, image.naturalHeight)
  } catch (error) {
    return error instanceof Error ? error.message : 'No se pudo leer la imagen.'
  } finally {
    URL.revokeObjectURL(url)
  }
}

export async function imageToPixels(file: File): Promise<number[]> {
  const validationError = validateImage(file)
  if (validationError) throw new Error(validationError)

  const url = URL.createObjectURL(file)
  try {
    const image = await decodeImage(url)
    const sizeError = dimensionError(image.naturalWidth, image.naturalHeight)
    if (sizeError) throw new Error(sizeError)
    const canvas = document.createElement('canvas')
    canvas.width = IMAGE_WIDTH
    canvas.height = IMAGE_HEIGHT
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) throw new Error('El navegador no pudo procesar la imagen.')
    context.drawImage(image, 0, 0)
    const rgba = context.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT).data
    return rgbaToPixels(rgba)
  } catch (error) {
    if (error instanceof Error) throw error
    throw new Error('No se pudo leer la imagen.')
  } finally {
    URL.revokeObjectURL(url)
  }
}
