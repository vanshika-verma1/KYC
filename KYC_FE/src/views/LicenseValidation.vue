<template>
  <div class="max-w-4xl mx-auto">
    <div class="card">
      <h2 class="text-3xl font-bold text-gray-900 mb-2">License Validation</h2>
      <p class="text-gray-600 mb-8">
        Please upload clear photos of both sides of your government-issued ID card.
      </p>

      <!-- Front Image Upload -->
      <div class="mb-8">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Front Side</h3>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-500 transition-colors">
          <input
            ref="frontInput"
            type="file"
            accept="image/*"
            @change="onFrontImageChange"
            class="hidden"
          />
          <div v-if="!frontImage" class="space-y-4">
            <div class="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
              <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p class="text-gray-600 mb-2">Upload front side of your ID</p>
              <button @click="$refs.frontInput.click()" class="btn-primary">
                Choose File
              </button>
            </div>
          </div>
          <div v-else class="space-y-4">
            <img :src="frontImage" alt="Front side" class="max-w-full h-48 object-contain mx-auto rounded-lg" />
            <div class="flex justify-center space-x-4">
              <button @click="$refs.frontInput.click()" class="btn-secondary">
                Change Image
              </button>
              <button @click="removeFrontImage" class="btn-secondary text-red-600 hover:bg-red-50">
                Remove
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Back Image Upload -->
      <div class="mb-8">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Back Side</h3>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-500 transition-colors">
          <input
            ref="backInput"
            type="file"
            accept="image/*"
            @change="onBackImageChange"
            class="hidden"
          />
          <div v-if="!backImage" class="space-y-4">
            <div class="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
              <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p class="text-gray-600 mb-2">Upload back side of your ID</p>
              <button @click="$refs.backInput.click()" class="btn-primary">
                Choose File
              </button>
            </div>
          </div>
          <div v-else class="space-y-4">
            <img :src="backImage" alt="Back side" class="max-w-full h-48 object-contain mx-auto rounded-lg" />
            <div class="flex justify-center space-x-4">
              <button @click="$refs.backInput.click()" class="btn-secondary">
                Change Image
              </button>
              <button @click="removeBackImage" class="btn-secondary text-red-600 hover:bg-red-50">
                Remove
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Submit Button -->
      <div class="flex justify-end">
        <button
          @click="validateLicense"
          :disabled="!frontImage || !backImage || isLoading"
          class="btn-primary disabled:opacity-50 disabled:cursor-not-allowed px-8 py-3"
        >
          <span v-if="isLoading" class="flex items-center">
            <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Validating...
          </span>
          <span v-else>Validate License</span>
        </button>
      </div>

      <!-- Error Message -->
      <div v-if="error" class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex">
          <svg class="w-5 h-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
          </svg>
          <div class="ml-3">
            <p class="text-sm text-red-800">{{ error }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'

const router = useRouter()
const store = useKycStore()

const frontInput = ref<HTMLInputElement>()
const backInput = ref<HTMLInputElement>()
const frontImage = ref<string>('')
const backImage = ref<string>('')
const isLoading = ref(false)
const error = ref('')

const frontFile = ref<File | null>(null)
const backFile = ref<File | null>(null)

const onFrontImageChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    frontFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      frontImage.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
  }
}

const onBackImageChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    backFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      backImage.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
  }
}

const removeFrontImage = () => {
  frontImage.value = ''
  frontFile.value = null
  if (frontInput.value) {
    frontInput.value.value = ''
  }
}

const removeBackImage = () => {
  backImage.value = ''
  backFile.value = null
  if (backInput.value) {
    backInput.value.value = ''
  }
}

const validateLicense = async () => {
  if (!frontFile.value || !backFile.value) return

  isLoading.value = true
  error.value = ''

  try {
    const formData = new FormData()
    formData.append('front_image', frontFile.value)
    formData.append('back_image', backFile.value)

    const response = await fetch('/license/validate_license', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    // Store license validation result and front image file
    store.setLicenseResult(result)
    if (frontFile.value) {
      store.setFrontImageFile(frontFile.value)
    }

    // Navigate to next step
    router.push('/liveness')
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'An error occurred during validation'
  } finally {
    isLoading.value = false
  }
}
</script>