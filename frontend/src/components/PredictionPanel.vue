<script setup lang="ts">
import { computed } from 'vue'
import { CLASS_DETAILS, CLASS_ORDER, type ImageResult } from '../types/model'

const props = defineProps<{ item: ImageResult }>()
const detail = computed(() => props.item.result ? CLASS_DETAILS[props.item.result.prediction.class] : null)

function percent(value: number): string {
  return `${(value * 100).toFixed(1)} %`
}
</script>

<template>
  <section class="result-panel" aria-live="polite">
    <div class="result-head">
      <span class="eyebrow">Resultado del modelo</span>
      <span v-if="item.result" class="model-tag">v{{ item.result.model_version }}</span>
    </div>
    <div v-if="item.status === 'processing'" class="result-placeholder">
      <span class="spinner" aria-hidden="true" /> Procesando imagen…
    </div>
    <div v-else-if="item.status === 'error'" class="error-box" role="alert">{{ item.error }}</div>
    <div v-else-if="item.result && detail">
      <div class="prediction-banner" :style="{ '--class-color': detail.color }">
        <span class="prediction-symbol" aria-hidden="true">●</span>
        <div>
          <span>Clasificación estimada</span>
          <h3>{{ detail.label }}</h3>
        </div>
        <strong>{{ percent(item.result.prediction.confidence) }}</strong>
      </div>
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
                width: percent(item.result.probabilities[name]),
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
