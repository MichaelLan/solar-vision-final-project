<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import ImageUpload from '../components/ImageUpload.vue'
import PredictionPanel from '../components/PredictionPanel.vue'
import { useClassifier } from '../composables/useClassifier'
import { getModelInfo } from '../services/api'
import { CLASS_DETAILS, CLASS_ORDER, type ImageResult } from '../types/model'

const { items, activeId, activeItem, busy, notice, ready, invalid, checking, completed, counts, addFiles, classifyAll, clear } = useClassifier()
const modelStatus = ref<'checking' | 'online' | 'offline'>('checking')
const pending = computed(() => ready.value.length)

const STATUS_LABELS: Record<ImageResult['status'], string> = {
  checking: 'comprobando',
  ready: 'pendiente',
  processing: 'procesando',
  done: 'clasificada',
  error: 'error',
}

function statusLabel(item: ImageResult): string {
  return item.rejected ? 'no válida' : STATUS_LABELS[item.status]
}

function plural(count: number, singular: string, pluralForm: string): string {
  return `${count} ${count === 1 ? singular : pluralForm}`
}

const analyzeLabel = computed(() => {
  if (busy.value) return 'Analizando…'
  if (checking.value) return 'Comprobando imágenes…'
  if (pending.value) return `Clasificar ${plural(pending.value, 'imagen', 'imágenes')}`
  return completed.value.length ? 'Lote analizado' : 'Sin imágenes válidas'
})

function goToWorkspace() {
  document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  document.getElementById('image-input')?.focus({ preventScroll: true })
}

onMounted(async () => {
  try {
    await getModelInfo()
    modelStatus.value = 'online'
  } catch {
    modelStatus.value = 'offline'
  }
})
</script>

<template>
  <div class="site-shell">
    <header class="site-header">
      <a class="brand" href="#" aria-label="Solar Vision, inicio">
        <span class="brand-mark" aria-hidden="true"><span /><span /><span /><span /></span>
        <span>solar<span class="brand-accent">vision</span></span>
      </a>
      <div class="header-right">
        <span class="project-pill">Proyecto de visión por computadora</span>
        <span class="status" :class="modelStatus">
          <span class="status-dot" />
          {{ modelStatus === 'online' ? 'Modelo disponible' : modelStatus === 'checking' ? 'Conectando…' : 'API no disponible' }}
        </span>
      </div>
    </header>

    <main>
      <section class="hero">
        <div class="hero-copy">
          <span class="eyebrow hero-eyebrow"><span class="eyebrow-line" /> INSPECCIÓN FOTOVOLTAICA · TERMOGRAFÍA</span>
          <h1>Una lectura más clara de tus <em>imágenes térmicas.</em></h1>
          <p>Clasifica patrones en módulos fotovoltaicos con un modelo de visión por computadora. Sube una imagen o analiza un lote completo en segundos.</p>
          <a class="button button-primary hero-cta" href="#workspace" @click.prevent="goToWorkspace">Analizar imágenes <span aria-hidden="true">↓</span></a>
          <div class="hero-meta"><span>04 categorías</span><span class="meta-divider" /><span>Imágenes de 24 × 40 px</span><span class="meta-divider" /><span>Resultados por imagen</span></div>
        </div>
        <div class="hero-art" aria-hidden="true">
          <div class="art-grid">
            <div v-for="n in 60" :key="n" class="art-cell" :style="{ '--n': n }" />
          </div>
          <div class="art-scanline" />
          <div class="art-label art-label-top"><span class="art-pulse" /> ANÁLISIS TÉRMICO</div>
          <div class="art-label art-label-bottom">24 × 40 <span>PÍXELES</span></div>
        </div>
      </section>

      <section id="workspace" class="workspace" aria-labelledby="workspace-title">
        <div class="section-heading">
          <div>
            <span class="eyebrow">01 / CLASIFICADOR</span>
            <h2 id="workspace-title">Analiza tus imágenes</h2>
            <p>Utiliza imágenes térmicas del mismo formato que el conjunto de entrenamiento.</p>
          </div>
          <button v-if="items.length" class="text-button" type="button" :disabled="busy" @click="clear">Limpiar lote ↗</button>
        </div>

        <ImageUpload @select="addFiles" />
        <p v-if="notice" class="notice" role="status">{{ notice }}</p>

        <div v-if="items.length" class="analysis-layout">
          <section class="image-card" aria-label="Vista previa y archivos">
            <div class="card-top"><span class="eyebrow">VISTA PREVIA</span><span>{{ items.length }} {{ items.length === 1 ? 'imagen' : 'imágenes' }}</span></div>
            <div v-if="activeItem" class="preview-frame">
              <img :src="activeItem.previewUrl" :alt="`Vista previa de ${activeItem.file.name}`" />
            </div>
            <div v-if="activeItem" class="preview-caption"><span class="filename" :title="activeItem.file.name">{{ activeItem.file.name }}</span><span>{{ (activeItem.file.size / 1024).toFixed(1) }} KB</span></div>
            <div class="analysis-actions">
              <button class="button button-primary analyze-button" type="button" :disabled="busy || checking || !pending" @click="classifyAll">
                <span v-if="busy || checking" class="spinner spinner-light" aria-hidden="true" />
                {{ analyzeLabel }}
                <span v-if="!busy && !checking" aria-hidden="true">↗</span>
              </button>
              <p v-if="invalid.length" class="invalid-note" role="status">
                {{ plural(invalid.length, 'imagen no válida', 'imágenes no válidas') }}: no se {{ invalid.length === 1 ? 'enviará' : 'enviarán' }} al modelo. Selecciónala{{ invalid.length === 1 ? '' : 's' }} para ver el motivo.
              </p>
            </div>
            <div v-if="items.length > 1" class="file-list">
              <button
                v-for="(item, index) in items"
                :key="item.id"
                class="file-row"
                :class="{ selected: item.id === activeId }"
                type="button"
                :aria-current="item.id === activeId ? 'true' : undefined"
                @click="activeId = item.id"
              >
                <span class="file-index">{{ String(index + 1).padStart(2, '0') }}</span>
                <span class="file-name" :title="item.file.name">{{ item.file.name }}</span>
                <span class="file-status" :class="item.status">
                  <span aria-hidden="true">{{ item.status === 'done' ? '✓' : item.status === 'error' ? '!' : item.status === 'processing' || item.status === 'checking' ? '…' : '○' }}</span>
                  <span class="visually-hidden">Estado: {{ statusLabel(item) }}</span>
                </span>
              </button>
            </div>
          </section>
          <PredictionPanel v-if="activeItem" :item="activeItem" />
        </div>

        <section v-if="items.length > 1 && completed.length" class="summary-card" aria-labelledby="summary-title">
          <div class="summary-heading"><div><span class="eyebrow">RESUMEN DEL LOTE</span><h3 id="summary-title">{{ plural(completed.length, 'imagen clasificada', 'imágenes clasificadas') }}</h3></div><span>Distribución de resultados</span></div>
          <div class="summary-grid">
            <div v-for="name in CLASS_ORDER" :key="name" class="summary-item">
              <span class="summary-dot" :style="{ backgroundColor: CLASS_DETAILS[name].color }" />
              <strong>{{ counts[name] }}</strong>
              <span>{{ CLASS_DETAILS[name].label }}</span>
            </div>
          </div>
        </section>
      </section>

      <section class="about-strip">
        <div><span class="eyebrow">02 / ALCANCE</span><h2>Cuatro patrones.<br /><em>Una primera orientación.</em></h2></div>
        <div class="about-grid">
          <div v-for="(name, index) in CLASS_ORDER" :key="name" class="about-item">
            <span class="about-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <span class="about-indicator" :style="{ backgroundColor: CLASS_DETAILS[name].color }" />
            <h3>{{ CLASS_DETAILS[name].label }}</h3>
            <p>{{ CLASS_DETAILS[name].description }}</p>
          </div>
        </div>
      </section>

      <p class="disclaimer">Esta herramienta es una demostración académica de clasificación de patrones térmicos. Sus resultados no sustituyen una inspección técnica ni constituyen un diagnóstico eléctrico.</p>
    </main>

    <footer class="site-footer"><span>solar<span class="brand-accent">vision</span></span><span>Proyecto final · Visión por Computadora</span></footer>
  </div>
</template>
