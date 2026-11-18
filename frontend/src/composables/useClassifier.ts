import { computed, onBeforeUnmount, ref } from 'vue'
import { predict } from '../services/api'
import { imageToPixels, MAX_BATCH_SIZE, validateImage } from '../services/image'
import { CLASS_ORDER, type ClassName, type ImageResult } from '../types/model'

export function useClassifier() {
  const items = ref<ImageResult[]>([])
  const activeId = ref<string | null>(null)
  const busy = ref(false)
  const notice = ref('')

  const activeItem = computed(() => items.value.find((item) => item.id === activeId.value) ?? null)
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
      const item: ImageResult = {
        id: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
        status: error ? 'error' : 'ready',
        ...(error ? { error } : {}),
      }
      items.value.push(item)
      activeId.value = item.id
    }
  }

  async function classifyAll() {
    if (busy.value) return
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

  return { items, activeId, activeItem, busy, notice, completed, counts, addFiles, classifyAll, clear }
}
