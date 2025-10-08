<template>
  <div class="max-w-5xl mx-auto">
    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-8">
      <div class="text-center mb-8">
        <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h2 class="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-2">
          License Validation
        </h2>
        <p class="text-gray-600 text-lg">
          Please upload clear photos of both sides of your government-issued ID card.
        </p>
      </div>

      <!-- Upload Sections -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

        <!-- Front Image Upload -->
        <div class="space-y-4">
          <div class="text-center">
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Front Side</h3>
            <p class="text-sm text-gray-500">Upload the front of your ID card</p>
          </div>
          <div class="relative group">
            <div :class="[
              'border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer',
              'hover:scale-[1.02] hover:shadow-lg',
              frontImage ? 'border-green-300 bg-green-50/50' : 'border-gray-300 hover:border-blue-400 bg-gray-50/50'
            ]" @click="triggerFrontInput">
              <input
                ref="frontInput"
                type="file"
                accept="image/*"
                @change="onFrontImageChange"
                class="hidden"
              />
              <div v-if="!frontImage" class="space-y-4">
                <div class="mx-auto w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg class="w-10 h-10 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-600 mb-3">Drag & drop or click to upload</p>
                  <button class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
                    Choose Front Image
                  </button>
                </div>
              </div>
              <div v-else class="space-y-4">
                <div class="relative">
                  <img :src="frontImage" alt="Front side" class="w-full h-48 object-contain rounded-xl shadow-md" />
                  <div class="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-xl"></div>
                  <div class="absolute top-2 right-2">
                    <div class="bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                      ✓ Uploaded
                    </div>
                  </div>
                </div>
                <div class="flex justify-center space-x-3">
                  <button @click.stop="triggerFrontInput" class="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors">
                    Change Image
                  </button>
                  <button @click.stop="removeFrontImage" class="bg-red-50 hover:bg-red-100 text-red-600 font-medium py-2 px-4 rounded-lg transition-colors">
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Back Image Upload -->
        <div class="space-y-4">
          <div class="text-center">
            <h3 class="text-xl font-semibold text-gray-900 mb-2">Back Side</h3>
            <p class="text-sm text-gray-500">Upload the back of your ID card</p>
          </div>
          <div class="relative group">
            <div :class="[
              'border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer',
              'hover:scale-[1.02] hover:shadow-lg',
              backImage ? 'border-green-300 bg-green-50/50' : 'border-gray-300 hover:border-blue-400 bg-gray-50/50'
            ]" @click="triggerBackInput">
              <input
                ref="backInput"
                type="file"
                accept="image/*"
                @change="onBackImageChange"
                class="hidden"
              />
              <div v-if="!backImage" class="space-y-4">
                <div class="mx-auto w-20 h-20 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg class="w-10 h-10 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-600 mb-3">Drag & drop or click to upload</p>
                  <button class="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-xl transition-all">
                    Choose Back Image
                  </button>
                </div>
              </div>
              <div v-else class="space-y-4">
                <div class="relative">
                  <img :src="backImage" alt="Back side" class="w-full h-48 object-contain rounded-xl shadow-md" />
                  <div class="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-xl"></div>
                  <div class="absolute top-2 right-2">
                    <div class="bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                      ✓ Uploaded
                    </div>
                  </div>
                </div>
                <div class="flex justify-center space-x-3">
                  <button @click.stop="triggerBackInput" class="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors">
                    Change Image
                  </button>
                  <button @click.stop="removeBackImage" class="bg-red-50 hover:bg-red-100 text-red-600 font-medium py-2 px-4 rounded-lg transition-colors">
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Submit Button -->
      <div class="text-center pt-6">
        <button
          @click="validateLicense"
          :disabled="!frontImage || !backImage || isLoading"
          :class="[
            'relative px-12 py-4 rounded-2xl font-bold text-lg transition-all duration-300 transform',
            'disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none',
            frontImage && backImage && !isLoading
              ? 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white shadow-lg hover:shadow-xl hover:scale-105'
              : 'bg-gradient-to-r from-gray-400 to-gray-500 text-white shadow-md'
          ]"
        >
          <span v-if="isLoading" class="flex items-center justify-center">
            <svg class="animate-spin -ml-1 mr-3 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Validating License...
          </span>
          <span v-else class="flex items-center justify-center">
            <svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Validate & Continue
          </span>
        </button>

        <!-- Progress indicator for both images -->
        <div class="flex justify-center space-x-4 mt-4">
          <div class="flex items-center space-x-2 text-sm">
            <div :class="frontImage ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="frontImage ? 'text-green-600 font-medium' : 'text-gray-500'">Front Image</span>
          </div>
          <div class="flex items-center space-x-2 text-sm">
            <div :class="backImage ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="backImage ? 'text-green-600 font-medium' : 'text-gray-500'">Back Image</span>
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
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
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

const triggerFrontInput = () => {
  frontInput.value?.click()
}

const triggerBackInput = () => {
  backInput.value?.click()
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
    router.push('/selfie')
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'An error occurred during validation'
  } finally {
    isLoading.value = false
  }
}
</script>