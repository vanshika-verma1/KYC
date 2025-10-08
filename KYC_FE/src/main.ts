import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { createPinia } from 'pinia'
import App from '@/App.vue'
import './style.css'

// Import views
import LicenseValidation from '@/views/LicenseValidation.vue'
import LivenessDetection from '@/views/LivenessDetection.vue'
import SelfieValidation from '@/views/SelfieValidation.vue'
import ResultsView from '@/views/ResultsView.vue'

const routes = [
  { path: '/', redirect: '/license' },
  { path: '/license', component: LicenseValidation },
  { path: '/liveness', component: LivenessDetection },
  { path: '/selfie', component: SelfieValidation },
  { path: '/results', component: ResultsView }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')