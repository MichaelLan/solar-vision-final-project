<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{ select: [files: File[]] }>()
const fileInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)

function accept(files: FileList | null) {
  if (files?.length) emit('select', Array.from(files))
  if (fileInput.value) fileInput.value.value = ''
}

function onDrop(event: DragEvent) {
  dragging.value = false
  accept(event.dataTransfer?.files ?? null)
}
</script>

<template>
  <div
    class="upload-zone"
    :class="{ 'is-dragging': dragging }"
    @dragover.prevent="dragging = true"
    @dragleave.prevent="dragging = false"
    @drop.prevent="onDrop"
  >
    <input
      ref="fileInput"
      id="image-input"
      class="visually-hidden"
      type="file"
      accept="image/png,image/jpeg,image/webp"
      multiple
      @change="accept(($event.target as HTMLInputElement).files)"
    />
    <div class="upload-icon" aria-hidden="true">↥</div>
    <div>
      <strong>Arrastra imágenes térmicas aquí</strong>
      <p>o selecciona archivos desde tu equipo</p>
    </div>
    <label class="button button-primary" for="image-input">Seleccionar imágenes</label>
    <small>PNG, JPEG o WebP · 24 × 40 px · hasta 5 MB por imagen · máximo 20</small>
  </div>
</template>
