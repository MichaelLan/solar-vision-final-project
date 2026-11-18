<script setup lang="ts">
import { computed } from 'vue'
import { CLASS_DETAILS, CLASS_ORDER, type ImageResult } from '../types/model'

const props = defineProps<{ item: ImageResult }>()
const detail = computed(() => props.item.result ? CLASS_DETAILS[props.item.result.prediction.class] : null)

/** Below this top-class probability the result is flagged as uncertain (not a calibrated confidence). */
const LOW_PROBABILITY = 0.5
const lowProbability = computed(() => (props.item.result?.prediction.confidence ?? 1) < LOW_PROBABILITY)

const percentFormat = new Intl.NumberFormat('es', { minimumFractionDigits: 1, maximumFractionDigits: 1 })

/** Formatted text for display in Spanish style, e.g. "24,8 %". */
function percent(value: number): string {
  return `${percentFormat.format(value * 100)} %`
}

/** Valid CSS width for the probability bar, e.g. "24.8%" (always a dot, never a locale comma). */
function barWidth(value: number): string {
  const clamped = Math.min(Math.max(Number.isFinite(value) ? value : 0, 0), 1)
  return `${clamped * 100}%`
}

/** Short text for the live region, so screen readers hear only the outcome. */
const announcement = computed(() => {
  const { item } = props
  if (item.status === 'processing') return 'Procesando imagen…'
  if (item.status === 'error') return `Error: ${item.error ?? ''}`
  if (item.result && detail.value) {
    const base = `Resultado: ${detail.value.label}, ${percent(item.result.prediction.confidence)}.`
    return lowProbability.value ? `${base} Probabilidad máxima baja.` : base
  }
  return ''
})
</script>

<template>
  <section class="result-panel" aria-labelledby="result-title">
    <p class="visually-hidden" aria-live="polite" aria-atomic="true">{{ announcement }}</p>
    <div class="result-head">
      <span id="result-title" class="eyebrow">Resultado del modelo</span>
      <span v-if="item.result" class="model-tag">v{{ item.result.model_version }}</span>
    </div>
    <div v-if="item.status === 'processing'" class="result-placeholder">
      <span class="spinner" aria-hidden="true" /> Procesando imagen…
    </div>
    <div v-else-if="item.status === 'checking'" class="result-placeholder">
      <span class="spinner" aria-hidden="true" /> Comprobando la imagen…
    </div>
    <div v-else-if="item.status === 'error'" class="error-box">
      <strong>{{ item.rejected ? 'Imagen no válida' : 'No se pudo clasificar la imagen' }}</strong>
      <p>{{ item.error }}</p>
      <small v-if="item.errorDetail" class="error-detail">Detalle técnico: {{ item.errorDetail }}</small>
    </div>
    <div v-else-if="item.result && detail">
      <div class="prediction-banner" :style="{ '--class-color': detail.color, '--class-ink': detail.ink }">
        <span class="prediction-symbol" aria-hidden="true">●</span>
        <div>
          <span>Clasificación estimada</span>
          <h3>{{ detail.label }}</h3>
        </div>
        <strong>{{ percent(item.result.prediction.confidence) }}</strong>
      </div>
      <p v-if="lowProbability" class="low-probability-note">
        Resultado incierto: la probabilidad máxima es baja; interprete el resultado con cautela.
      </p>
      <p class="result-description">{{ detail.description }}</p>
      <h4>Probabilidad por categoría</h4>
      <div class="probability-list">
        <div v-for="name in CLASS_ORDER" :key="name" class="probability-row">
          <span class="probability-name">{{ CLASS_DETAILS[name].label }}</span>
          <div
            class="probability-track"
            role="progressbar"
            :aria-label="CLASS_DETAILS[name].label"
            :aria-valuenow="Math.round(item.result.probabilities[name] * 100)"
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <div
              class="probability-fill"
              :style="{
                width: barWidth(item.result.probabilities[name]),
                backgroundColor: CLASS_DETAILS[name].color,
              }"
            />
          </div>
          <strong>{{ percent(item.result.probabilities[name]) }}</strong>
        </div>
      </div>
    </div>
    <div v-else class="result-placeholder">Selecciona una imagen y ejecuta la clasificación.</div>
  </section>
</template>
