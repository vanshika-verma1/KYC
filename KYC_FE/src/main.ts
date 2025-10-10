import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { createPinia } from 'pinia'
import App from './App.vue'
import './style.css'

// Import views
import LicenseValidation from './views/LicenseValidation.vue'
import LivenessDetection from './views/LivenessDetection.vue'
import SelfieValidation from './views/SelfieValidation.vue'
import ResultsView from './views/ResultsView.vue'

// Verification state management
const verificationState = {
  currentStep: 'license',
  completedSteps: new Set<string>(),
  canAccess: (route: string): boolean => {
    switch (route) {
      case '/license':
        return true // Always accessible as starting point
      case '/liveness':
        return verificationState.completedSteps.has('license')
      case '/selfie':
        return verificationState.completedSteps.has('license')
      case '/results':
        return verificationState.completedSteps.has('license') &&
               verificationState.completedSteps.has('selfie')
      default:
        return false
    }
  },
  completeStep: (step: string) => {
    verificationState.completedSteps.add(step)
  },
  reset: () => {
    verificationState.currentStep = 'license'
    verificationState.completedSteps.clear()
  }
}

const routes = [
  {
    path: '/',
    redirect: '/license'
  },
  {
    path: '/license',
    component: LicenseValidation,
    meta: { step: 'license', requiresAuth: false }
  },
  {
    path: '/liveness',
    component: LivenessDetection,
    meta: { step: 'liveness', requiresAuth: true }
  },
  {
    path: '/selfie',
    component: SelfieValidation,
    meta: { step: 'selfie', requiresAuth: true }
  },
  {
    path: '/results',
    component: ResultsView,
    meta: { step: 'results', requiresAuth: true }
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/license' // Catch all other routes and redirect to license
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Route guard to control navigation
router.beforeEach((to, from, next) => {
  // Prevent direct access to protected routes
  if (to.meta.requiresAuth && !verificationState.canAccess(to.path)) {
    next('/license')
    return
  }

  // Allow access to license (starting point)
  if (to.path === '/license') {
    next()
    return
  }

  // For other routes, check if user has completed required steps
  if (!verificationState.canAccess(to.path)) {
    next('/license')
    return
  }

  next()
})

// Disable browser back/forward functionality
let isNavigationBlocked = false

// Override browser history methods
const originalPushState = history.pushState
const originalReplaceState = history.replaceState

history.pushState = function(state: any, title: string, url?: string | URL | null) {
  if (!isNavigationBlocked) {
    originalPushState.call(history, state, title, url)
  }
}

history.replaceState = function(state: any, title: string, url?: string | URL | null) {
  if (!isNavigationBlocked) {
    originalReplaceState.call(history, state, title, url)
  }
}

// Prevent back/forward navigation
window.addEventListener('popstate', (event) => {
  if (isNavigationBlocked) {
    // Push the current state back to prevent navigation
    history.pushState(null, '', window.location.href)
    return
  }
})

// Block navigation after router is ready
router.isReady().then(() => {
  isNavigationBlocked = true

  // Push multiple states to make back button ineffective
  for (let i = 0; i < 10; i++) {
    history.pushState(null, '', window.location.href)
  }
})

// Expose verification state for components to use
;(window as any).verificationState = verificationState

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')