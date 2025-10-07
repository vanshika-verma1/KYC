<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-4">
          <div class="flex items-center">
            <h1 class="text-2xl font-bold text-gray-900">KYC Verification</h1>
          </div>
          <div class="flex items-center space-x-4">
            <div class="text-sm text-gray-500">
              Step {{ currentStep }} of {{ totalSteps }}
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Progress Bar -->
    <div class="bg-white border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="py-4">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-8">
              <div
                v-for="(step, index) in steps"
                :key="index"
                class="flex items-center"
              >
                <div
                  :class="[
                    'step-indicator',
                    step.status === 'completed' && 'step-completed',
                    step.status === 'active' && 'step-active',
                    step.status === 'pending' && 'step-inactive'
                  ]"
                >
                  {{ index + 1 }}
                </div>
                <div class="ml-3">
                  <p class="text-sm font-medium text-gray-900">{{ step.title }}</p>
                  <p class="text-xs text-gray-500">{{ step.description }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <router-view />
    </main>
  </div>
</template>

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
    title: 'Liveness Detection',
    description: 'Verify live presence',
    status: computed(() => getStepStatus('/liveness')),
    path: '/liveness'
  },
  {
    title: 'Selfie Validation',
    description: 'Match your face',
    status: computed(() => getStepStatus('/selfie')),
    path: '/selfie'
  }
]

function getStepStatus(path: string) {
  const currentPath = router.currentRoute.value.path
  if (currentPath === path) return 'active'
  if (currentPath === '/results') return 'completed'
  return 'pending'
}

const currentStep = computed(() => {
  const currentPath = router.currentRoute.value.path
  const stepIndex = steps.findIndex(step => step.path === currentPath)
  return stepIndex >= 0 ? stepIndex + 1 : 1
})

const totalSteps = computed(() => steps.length)
</script>