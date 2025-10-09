<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'
import {environments} from "@/env.ts"

const router = useRouter()
const store = useKycStore()


// Template refs
const fileInput = ref<HTMLInputElement>()
const canvasElement = ref<HTMLCanvasElement>()

// Reactive state
const selfieImage = ref<string>('')
const selfieFile = ref<File | null>(null)
const isLoading = ref(false)
const error = ref('')

// File upload functions
const triggerFileInput = () => {
  fileInput.value?.click()
}

const onFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    selfieFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      selfieImage.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
  }
}

const removeImage = () => {
  selfieImage.value = ''
  selfieFile.value = null
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

// API validation function
const validateSelfie = async () => {
  if (!selfieFile.value || !store.frontImageFile) return

  isLoading.value = true
  error.value = ''

  try {
    const formData = new FormData()
    formData.append('license', selfieFile.value)  // Selfie image
    formData.append('selfie', store.frontImageFile)  // License front image

    const response = await fetch(`${environments.baseurl}/selfie/validate_selfie`, {
      method: 'POST',
      body: formData
      // Don't set Content-Type header - let the browser set it with proper boundary for FormData
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Server response:', response.status, errorText)
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
    }

    const result = await response.json()

    // Store selfie validation result
    store.setSelfieResult(result)

    // Navigate to results page
    router.push('/results')
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'An error occurred during validation'
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="max-w-4xl mx-auto">
    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
        <h2 class="text-2xl font-semibold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-2">
          Selfie Validation
        </h2>
        <p class="text-gray-600 text-md">
          Please take a clear selfie or upload a photo of yourself for identity verification.
        </p>
      </div>

      <!-- Image Upload/Capture Section -->
      <div class="max-w-2xl mx-auto mb-6">
        <div class="text-center mb-6">
          <h3 class="text-xl font-semibold text-gray-800 mb-1">Your Photo</h3>
          <p class="text-sm text-gray-500">Please upload clear photo of yourself</p>
        </div>

        <div class="relative group">
          <div :class="[
            'border-2 border-dashed rounded-2xl p-5 text-center transition-all duration-300 min-h-[280px] flex flex-col justify-center',
            'hover:scale-[1.02] hover:shadow-lg',
            selfieImage ? 'border-green-300 bg-green-50/50' : 'border-gray-300 hover:border-purple-400 bg-gray-50/50'
          ]">

            <!-- Empty State - Upload Button -->
            <div v-if="!selfieImage" class="space-y-2">
              <div class="mx-auto w-16 h-16 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                <svg class="w-8 h-8 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <div class="flex justify-center mt-4">
                  <button
                    @click="triggerFileInput"
                    :disabled="isLoading"
                    class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all disabled:opacity-50"
                  >
                    <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Upload Photo
                  </button>
                </div>
              </div>
            </div>

            <!-- Captured/Uploaded Image -->
            <div v-if="selfieImage" class="space-y-2 flex-1 flex flex-col justify-center">
              <div class="relative">
                <img :src="selfieImage" alt="Selfie" class="w-full h-40 object-contain rounded-xl" />
                <div class="absolute top-2 right-2">
                  <div class="bg-green-500 text-white text-xs font-medium px-2 py-1 rounded-full">
                    ✓ Ready
                  </div>
                </div>
              </div>
              <div class="flex justify-center space-x-3">
                <button @click="triggerFileInput" class="bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-gray-300">
                  Change Photo
                </button>
                <button @click="removeImage" class="bg-red-50 hover:bg-red-100 text-red-600 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-red-150">
                  Remove
                </button>
              </div>
            </div>

            <!-- Hidden File Input -->
            <input
              ref="fileInput"
              type="file"
              accept="image/*"
              @change="onFileChange"
              class="hidden"
            />
          </div>
        </div>
      </div>


      <!-- Validate Button -->
      <div class="text-center pt-2">
        <button
          @click="validateSelfie"
          :disabled="!selfieImage || !store.frontImageFile || isLoading"
          :class="[
            'relative px-8 py-3 rounded-xl font-medium text-base transition-all duration-300 transform',
            'disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none',
            selfieImage && store.frontImageFile && !isLoading
              ? 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
              : 'bg-gradient-to-r from-gray-400 to-gray-500 text-white shadow-md'
          ]"
        >
          <span v-if="isLoading" class="flex items-center justify-center">
            <svg class="animate-spin -ml-1 mr-3 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Validating Selfie...
          </span>
          <span v-else class="flex items-center justify-center">
            <svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Validate & Continue
          </span>
        </button>

        <!-- Progress indicator -->
        <div class="flex justify-center space-x-4 mt-4">
          <div class="flex items-center space-x-2 text-xs">
            <div :class="selfieImage ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="selfieImage ? 'text-green-600 font-medium' : 'text-gray-500'">Selfie Photo</span>
          </div>
          <div class="flex items-center space-x-2 text-xs">
            <div :class="store.frontImageFile ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="store.frontImageFile ? 'text-green-600 font-medium' : 'text-gray-500'">ID Document</span>
          </div>
        </div>
      </div>

      <!-- Error Message -->
      <div v-if="error" class="mt-8 p-6 bg-red-50/80 backdrop-blur-sm border border-red-200 rounded-2xl">
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <svg class="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="ml-3">
            <h3 class="text-sm font-semibold text-red-800">Validation Error</h3>
            <p class="mt-1 text-sm text-red-700">{{ error }}</p>
          </div>
        </div>
      </div>

      <!-- Hidden canvas for camera capture -->
      <canvas ref="canvasElement" class="hidden"></canvas>
    </div>
  </div>
</template>