import { computed, onBeforeUnmount, ref } from 'vue'
import { ApiError, predict } from '../services/api'
import { checkImage, imageToPixels, MAX_BATCH_SIZE, validateImage } from '../services/image'
import { CLASS_ORDER, type ClassName, type ImageResult } from '../types/model'

export function useClassifier() {
  const items = ref<ImageResult[]>([])
  const activeId = ref<string | null>(null)
  const busy = ref(false)
  const notice = ref('')

  const activeItem = computed(() => items.value.find((item) => item.id === activeId.value) ?? null)
  const ready = computed(() => items.value.filter((item) => item.status === 'ready'))
  const invalid = computed(() => items.value.filter((item) => item.rejected))
  const checking = computed(() => items.value.some((item) => item.status === 'checking'))
  const completed = computed(() => items.value.filter((item) => item.status === 'done'))
  const counts = computed(() => Object.fromEntries(
    CLASS_ORDER.map((name) => [name, completed.value.filter((item) => item.result?.prediction.class === name).length]),
  ) as Record<ClassName, number>)

  function addFiles(files: File[]) {
    notice.value = ''
    const remaining = MAX_BATCH_SIZE - items.value.length
    if (remaining <= 0) {
      notice.value = `El lote admite hasta ${MAX_BATCH_SIZE} imágenes. Limpia el lote para continuar.`
      return
    }
    if (files.length > remaining) notice.value = `Se añadieron ${remaining} imágenes; el máximo por lote es ${MAX_BATCH_SIZE}.`
    for (const file of files.slice(0, remaining)) {
      const error = validateImage(file)
      items.value.push({
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
        status: error ? 'error' : 'checking',
        ...(error ? { error, rejected: true } : {}),
      })
      const item = items.value[items.value.length - 1]
      activeId.value = item.id
      if (!error) void verify(item)
    }
  }

  async function verify(item: ImageResult) {
    const error = await checkImage(item.file)
    if (item.status !== 'checking') return
    if (error) {
      item.error = error
      item.rejected = true
      item.status = 'error'
    } else {
      item.status = 'ready'
    }
  }

  async function classifyAll() {
    if (busy.value || checking.value) return
    busy.value = true
    notice.value = ''
    try {
      for (const item of items.value) {
        if (item.status !== 'ready') continue
        item.status = 'processing'
        activeId.value = item.id
        try {
          const pixels = await imageToPixels(item.file)
          item.result = await predict(pixels)
          item.status = 'done'
        } catch (error) {
          item.error = error instanceof Error ? error.message : 'Ocurrió un error inesperado.'
          item.errorDetail = error instanceof ApiError ? error.detail : undefined
          item.status = 'error'
        }
      }
    } finally {
      busy.value = false
    }
  }

  function releasePreviews() {
    for (const item of items.value) URL.revokeObjectURL(item.previewUrl)
  }

  function clear() {
    if (busy.value) return
    releasePreviews()
    items.value = []
    activeId.value = null
    notice.value = ''
  }

  onBeforeUnmount(releasePreviews)

  return { items, activeId, activeItem, busy, notice, ready, invalid, checking, completed, counts, addFiles, classifyAll, clear }
}
