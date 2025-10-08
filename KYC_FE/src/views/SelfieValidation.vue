<template>
  <div class="max-w-5xl mx-auto">
    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-8">
      <div class="text-center mb-8">
        <div class="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </div>
        <h2 class="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-2">
          Selfie Validation
        </h2>
        <p class="text-gray-600 text-lg">
          Please take a clear selfie or upload a photo of yourself for identity verification.
        </p>
      </div>

      <!-- Image Upload/Capture Section -->
      <div class="max-w-2xl mx-auto mb-8">
        <div class="text-center mb-6">
          <h3 class="text-xl font-semibold text-gray-900 mb-2">Your Photo</h3>
          <p class="text-sm text-gray-500">Upload a photo or take one using your camera</p>
        </div>

        <div class="relative group">
          <div :class="[
            'border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300',
            'hover:scale-[1.02] hover:shadow-lg',
            selfieImage ? 'border-green-300 bg-green-50/50' : 'border-gray-300 hover:border-purple-400 bg-gray-50/50'
          ]">

            <!-- Camera Option -->
            <div v-if="!selfieImage && !showCamera" class="space-y-6">
              <div class="mx-auto w-20 h-20 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                <svg class="w-10 h-10 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>

              <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <button
                  @click="openCamera"
                  :disabled="isLoading"
                  class="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-xl transition-all disabled:opacity-50"
                >
                  <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Take Photo
                </button>

                <button
                  @click="triggerFileInput"
                  :disabled="isLoading"
                  class="bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 font-semibold py-3 px-6 rounded-xl shadow-lg hover:shadow-xl transition-all disabled:opacity-50"
                >
                  <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  Upload Photo
                </button>
              </div>

              <input
                ref="fileInput"
                type="file"
                accept="image/*"
                @change="onFileChange"
                class="hidden"
              />
            </div>

            <!-- Camera View -->
            <div v-if="showCamera && !selfieImage" class="space-y-4">
              <div class="relative bg-black rounded-xl overflow-hidden" style="aspect-ratio: 4/3; max-height: 400px;">
                <video
                  ref="videoElement"
                  class="w-full h-full object-cover"
                  autoplay
                  playsinline
                  muted
                ></video>
                <div class="absolute inset-0 border-2 border-purple-400 rounded-xl pointer-events-none"></div>

                <!-- Camera overlay guides -->
                <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div class="w-48 h-48 border-2 border-white/50 rounded-full"></div>
                </div>

                <div class="absolute bottom-4 left-1/2 transform -translate-x-1/2">
                  <div class="w-16 h-16 bg-white rounded-full border-4 border-purple-500 flex items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors shadow-lg" @click="capturePhoto">
                    <div class="w-6 h-6 bg-purple-500 rounded-full"></div>
                  </div>
                </div>

                <!-- Camera ready indicator -->
                <div class="absolute top-4 right-4">
                  <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                </div>
              </div>

              <div class="flex justify-center space-x-3">
                <button @click="closeCamera" class="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors">
                  Cancel
                </button>
              </div>
            </div>

            <!-- Captured/Uploaded Image -->
            <div v-if="selfieImage" class="space-y-4">
              <div class="relative">
                <img :src="selfieImage" alt="Selfie" class="w-full max-h-64 object-contain rounded-xl shadow-md" />
                <div class="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent rounded-xl"></div>
                <div class="absolute top-2 right-2">
                  <div class="bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                    ✓ Ready
                  </div>
                </div>
              </div>
              <div class="flex justify-center space-x-3">
                <button
                  v-if="!showCamera"
                  @click="openCamera"
                  class="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors"
                >
                  Retake Photo
                </button>
                <button @click="triggerFileInput" class="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium py-2 px-4 rounded-lg transition-colors">
                  Change Photo
                </button>
                <button @click="removeImage" class="bg-red-50 hover:bg-red-100 text-red-600 font-medium py-2 px-4 rounded-lg transition-colors">
                  Remove
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- License Reference -->
      <div v-if="store.frontImageFile" class="max-w-2xl mx-auto mb-8">
        <div class="bg-blue-50/80 backdrop-blur-sm border border-blue-200 rounded-2xl p-6">
          <div class="flex items-center space-x-3 mb-4">
            <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h3 class="font-semibold text-gray-900">ID Document Reference</h3>
              <p class="text-sm text-gray-600">We'll compare your selfie with your uploaded ID</p>
            </div>
          </div>
          <div class="text-center">
            <img :src="licenseImageUrl" alt="License front" class="w-full max-h-32 object-contain rounded-lg shadow-sm mx-auto" />
          </div>
        </div>
      </div>

      <!-- Validate Button -->
      <div class="text-center pt-6">
        <button
          @click="validateSelfie"
          :disabled="!selfieImage || !store.frontImageFile || isLoading"
          :class="[
            'relative px-12 py-4 rounded-2xl font-bold text-lg transition-all duration-300 transform',
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
          <div class="flex items-center space-x-2 text-sm">
            <div :class="selfieImage ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="selfieImage ? 'text-green-600 font-medium' : 'text-gray-500'">Selfie Photo</span>
          </div>
          <div class="flex items-center space-x-2 text-sm">
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

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'

const router = useRouter()
const store = useKycStore()

// Computed properties
const licenseImageUrl = computed(() => {
  if (store.frontImageFile) {
    return (window as any).URL?.createObjectURL
      ? (window as any).URL.createObjectURL(store.frontImageFile)
      : (window as any).webkitURL?.createObjectURL
        ? (window as any).webkitURL.createObjectURL(store.frontImageFile)
        : ''
  }
  return ''
})

// Template refs
const fileInput = ref<HTMLInputElement>()
const videoElement = ref<HTMLVideoElement>()
const canvasElement = ref<HTMLCanvasElement>()

// Reactive state
const selfieImage = ref<string>('')
const selfieFile = ref<File | null>(null)
const showCamera = ref(false)
const isLoading = ref(false)
const error = ref('')
const stream = ref<MediaStream | null>(null)

// Camera functions
const openCamera = async () => {
  try {
    // More comprehensive camera constraints
    const constraints = {
      video: {
        facingMode: 'user',
        width: { ideal: 1280, max: 1920 },
        height: { ideal: 720, max: 1080 }
      },
      audio: false
    }

    const mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
    stream.value = mediaStream

    if (videoElement.value) {
      videoElement.value.srcObject = mediaStream

      // Try to play video, but don't block on it
      try {
        await videoElement.value.play()
      } catch (playError) {
        console.warn('Auto-play failed, but video should still work:', playError)
        // Video might still work even if autoplay fails
      }
    }

    showCamera.value = true
    error.value = ''
  } catch (err) {
    console.error('Camera error:', err)
    if (err instanceof Error) {
      if (err.name === 'NotAllowedError') {
        error.value = 'Camera access denied. Please allow camera permissions and try again.'
      } else if (err.name === 'NotFoundError') {
        error.value = 'No camera found. Please upload a photo instead.'
      } else if (err.name === 'NotReadableError') {
        error.value = 'Camera is busy or unavailable. Please close other apps using the camera.'
      } else {
        error.value = `Camera error: ${err.message}. Please upload a photo instead.`
      }
    } else {
      error.value = 'Camera access failed. Please upload a photo instead.'
    }
  }
}

const closeCamera = () => {
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop())
    stream.value = null
  }
  showCamera.value = false
}

const capturePhoto = () => {
  if (!videoElement.value || !canvasElement.value) {
    error.value = 'Camera not ready. Please wait a moment and try again.'
    return
  }

  const video = videoElement.value
  const canvas = canvasElement.value
  const context = canvas.getContext('2d')

  if (!context) {
    error.value = 'Unable to capture image. Please try again.'
    return
  }

  // More lenient check - just ensure video is playing or has data
  if (video.readyState === video.HAVE_NOTHING) {
    error.value = 'Camera not ready. Please wait for the preview to load.'
    return
  }

  // Use actual video dimensions if available, otherwise use default
  const captureWidth = video.videoWidth || 1280
  const captureHeight = video.videoHeight || 720

  // Set canvas dimensions
  canvas.width = captureWidth
  canvas.height = captureHeight

  // Draw the video frame on the canvas
  context.drawImage(video, 0, 0, captureWidth, captureHeight)

  // Convert to blob and create file
  canvas.toBlob((blob) => {
    if (blob) {
      const file = new File([blob], `selfie-${Date.now()}.jpg`, { type: 'image/jpeg' })
      selfieFile.value = file

      // Create preview URL
      const reader = new FileReader()
      reader.onload = (e) => {
        selfieImage.value = e.target?.result as string
        showCamera.value = false
      }
      reader.onerror = () => {
        error.value = 'Error processing captured image. Please try again.'
      }
      reader.readAsDataURL(blob)
    } else {
      error.value = 'Failed to capture image. Please try again.'
    }
  }, 'image/jpeg', 0.9)

  closeCamera()
}

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

    const response = await fetch('/selfie/validate_selfie', {
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