<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const steps = [
  {
    title: 'License Validation',
    description: 'Upload ID images',
    status: computed(() => getStepStatus('/license')),
    path: '/license'
  },
  {
    title: 'Selfie Validation',
    description: 'Verify your identity',
    status: computed(() => getStepStatus('/selfie')),
    path: '/selfie'
  },
  {
    title: 'Liveness Detection',
    description: 'Verify live presence',
    subtext: '(pending)',
    status: computed(() => getStepStatus('/liveness')),
    path: '/liveness'
  },
  {
    title: 'Results',
    description: 'View verification results',
    status: computed(() => getStepStatus('/results')),
    path: '/results'
  }
]

function getStepStatus(path: string) {
  const currentPath = router.currentRoute.value.path
  if (currentPath === path) return 'active'
  if (currentPath === '/results') return 'completed'
  if (path === '/liveness') return 'pending' // Liveness is pending
  return 'pending'
}

const currentStep = computed(() => {
  const currentPath = router.currentRoute.value.path
  const stepIndex = steps.findIndex(step => step.path === currentPath)
  return stepIndex >= 0 ? stepIndex + 1 : 1
})

const totalSteps = computed(() => steps.length)
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
    <!-- Enhanced Header -->
    <header class="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-lg shadow-lg border-b border-white/20">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-3">
          <div class="flex items-center space-x-3">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h1 class="text-2xl font-semibold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                KYC Verification
              </h1>
              <p class="text-xs text-gray-500">Secure Identity Verification Platform</p>
            </div>
          </div>
          <div class="flex items-center space-x-8">
            <div class="hidden sm:flex items-center space-x-2 bg-blue-50 px-4 py-2 rounded-full">
              <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span class="text-xs font-medium text-blue-700">Step {{ currentStep }} of {{ totalSteps }}</span>
            </div>
            <div class="text-xs text-gray-400 bg-gray-100 px-3 py-1 rounded-full">
              Secure • Private • Fast
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Enhanced Progress Bar -->
    <div class="fixed top-16 mt-2 left-0 right-0 z-40 bg-gradient-to-r from-blue-600 via-blue-700 to-indigo-800 backdrop-blur-sm border-b border-blue-400/30 shadow-lg">
      <div class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="py-3">
          <div class="flex items-center justify-center">
            <div class="flex items-center space-x-3 sm:space-x-6">
              <div
                v-for="(step, index) in steps"
                :key="index"
                class="flex items-center"
              >
                <div
                  :class="[
                    'relative flex items-center justify-center w-10 h-10 rounded-full text-sm font-bold transition-all duration-300 transform',
                    step.status.value === 'completed' && 'bg-white text-green-600 shadow-lg scale-110',
                    step.status.value === 'active' && 'bg-white text-blue-600 shadow-xl scale-100 ring-4 ring-white/50',
                    step.status.value === 'pending' && 'bg-white/80 text-gray-600 backdrop-blur-sm'
                  ]"
                >
                  <span v-if="step.status.value === 'completed'" class="absolute inset-0 flex items-center justify-center">
                    <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                    </svg>
                  </span>
                  <span v-else>{{ index + 1 }}</span>
                </div>
                <div class="ml-2 sm:ml-3 hidden sm:block">
                  <p class="text-sm font-semibold text-white">{{ step.title }}</p>
                  <p class="text-xs text-blue-100">{{ step.description }}</p>
                  <p v-if="step.subtext" class="text-xs text-blue-200">{{ step.subtext }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <main class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-4 pt-44">
      <router-view />
    </main>
  </div>
</template>