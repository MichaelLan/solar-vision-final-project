import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import PredictionPanel from './PredictionPanel.vue'
import { CLASS_ORDER, type ImageResult, type PredictionResponse } from '../types/model'

function itemWith(result: PredictionResponse): ImageResult {
  return { id: '1', file: new File(['x'], 'a.png', { type: 'image/png' }), previewUrl: '', status: 'done', result }
}

const result: PredictionResponse = {
  prediction: { class: 'Cell', label: 'Cell', confidence: 0.6 },
  probabilities: { 'No-Anomaly': 0.248, Cell: 0.6, 'Cell-Multi': 0.1, Diode: 0.052 },
  model_version: '1.0.0',
}

describe('PredictionPanel', () => {
  it('renders probability bars with widths proportional to each probability', () => {
    const wrapper = mount(PredictionPanel, { props: { item: itemWith(result) } })
    const widths = wrapper.findAll('.probability-fill').map((bar) => (bar.element as HTMLElement).style.width)
    expect(widths).toHaveLength(4)
    widths.forEach((width, index) => {
      const expected = result.probabilities[CLASS_ORDER[index]] * 100
      expect(parseFloat(width)).toBeCloseTo(expected, 5)
      expect(width.endsWith('%')).toBe(true)
    })
    expect(widths).not.toContain('100%')
    expect(wrapper.text()).toContain('24,8 %')
    expect(wrapper.find('.prediction-banner strong').text()).toBe('60,0 %')
  })

  it('flags a low maximum probability without hiding the result', () => {
    const low = { ...result, prediction: { ...result.prediction, confidence: 0.42 } }
    const wrapper = mount(PredictionPanel, { props: { item: itemWith(low) } })
    expect(wrapper.find('.low-probability-note').text()).toMatch(/probabilidad máxima es baja/)
    expect(mount(PredictionPanel, { props: { item: itemWith(result) } }).find('.low-probability-note').exists()).toBe(false)
  })

  it('shows client-side validation errors with a clear reason', () => {
    const item: ImageResult = {
      id: '2', file: new File(['x'], 'b.png', { type: 'image/png' }), previewUrl: '',
      status: 'error', rejected: true, error: 'Tamaño no admitido: mide 50 × 50 px y debe medir 24 × 40 px.',
    }
    const wrapper = mount(PredictionPanel, { props: { item } })
    expect(wrapper.find('.error-box').text()).toContain('Imagen no válida')
    expect(wrapper.find('.error-box').text()).toContain('24 × 40')
  })
})
