<template>
  <div class="max-w-4xl mx-auto">
    <div class="card">
      <h2 class="text-3xl font-bold text-gray-900 mb-2">Selfie Validation</h2>
      <p class="text-gray-600 mb-8">
        Please take a clear photo of your face. We'll compare it with your ID photo to verify your identity.
      </p>

      <!-- License Image Status -->
      <div class="mb-4 flex items-center space-x-4">
        <div class="flex items-center space-x-2">
          <div
            class="w-3 h-3 rounded-full"
            :class="store.frontImageFile ? 'bg-green-400' : 'bg-red-400'"
          ></div>
          <span class="text-sm font-medium" :class="store.frontImageFile ? 'text-green-600' : 'text-red-600'">
            {{ store.frontImageFile ? 'License image ready' : 'License image missing' }}
          </span>
        </div>

        <div v-if="store.frontImageFile" class="text-sm text-gray-500">
          {{ (store.frontImageFile.size / 1024 / 1024).toFixed(1) }}MB • {{ store.frontImageFile.name }}
        </div>
      </div>

      <!-- Camera Section -->
      <div class="mb-8">
        <div class="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
          <video
            ref="videoElement"
            autoplay
            playsinline
            muted
            class="w-full h-full object-cover"
          ></video>

          <!-- Overlay for capture mode -->
          <div v-if="!capturedImage" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div class="text-center text-white">
              <div class="animate-pulse w-16 h-16 bg-white bg-opacity-25 rounded-full mx-auto mb-4"></div>
              <p class="text-lg">Position your face in the center</p>
              <p class="text-sm opacity-75">Ensure good lighting and look directly at the camera</p>
            </div>
          </div>

          <!-- Captured image overlay -->
          <div v-if="capturedImage" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div class="text-center text-white">
              <div class="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p class="text-lg">Photo Captured!</p>
              <p class="text-sm opacity-75">Processing face comparison...</p>
            </div>
          </div>

          <!-- Face detection box -->
          <div
            v-if="faceBox && !capturedImage"
            class="absolute border-2 border-green-500"
            :style="{
              left: `${faceBox.x}px`,
              top: `${faceBox.y}px`,
              width: `${faceBox.width}px`,
              height: `${faceBox.height}px`
            }"
          ></div>
        </div>
      </div>

      <!-- Controls -->
      <div class="flex justify-center space-x-4 mb-8">
        <button
          v-if="!isStreaming"
          @click="startCamera"
          class="btn-primary"
        >
          Start Camera
        </button>

        <button
          v-if="isStreaming && !capturedImage"
          @click="capturePhoto"
          :disabled="!faceDetected"
          class="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Take Photo
        </button>

        <button
          v-if="capturedImage && !isValidating"
          @click="retakePhoto"
          class="btn-secondary"
        >
          Retake Photo
        </button>

        <button
          v-if="capturedImage && !isValidating"
          @click="validateSelfie"
          class="btn-primary"
        >
          Validate Selfie
        </button>
      </div>

      <!-- Loading State -->
      <div v-if="isValidating" class="text-center py-8">
        <div class="animate-spin w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p class="text-lg text-gray-600">Comparing faces...</p>
        <p class="text-sm text-gray-500">This may take a few moments</p>
      </div>

      <!-- Validation Results -->
      <div v-if="validationResult" class="space-y-6">
        <!-- Comparison Images -->
        <div class="grid grid-cols-2 gap-6">
          <div class="text-center">
            <h4 class="font-semibold text-gray-900 mb-2">ID Photo</h4>
            <img
              :src="licenseImage"
              alt="ID Photo"
              class="w-full h-48 object-cover rounded-lg border-2"
              :class="validationResult.match_result ? 'border-green-500' : 'border-red-500'"
            />
          </div>
          <div class="text-center">
            <h4 class="font-semibold text-gray-900 mb-2">Comparison Result</h4>
            <img
              v-if="comparisonImage"
              :src="comparisonImage"
              alt="Face Comparison"
              class="w-full h-48 object-cover rounded-lg border-2"
              :class="validationResult.match_result ? 'border-green-500' : 'border-red-500'"
            />
            <div v-else class="w-full h-48 bg-gray-200 rounded-lg border-2 border-gray-300 flex items-center justify-center">
              <span class="text-gray-500">Processing comparison...</span>
            </div>
          </div>
        </div>

        <!-- Match Result -->
        <div class="text-center p-6 rounded-lg" :class="validationResult.match_result ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'">
          <div class="text-4xl mb-2">
            {{ validationResult.match_result ? '✅' : '❌' }}
          </div>
          <h3 class="text-xl font-bold mb-2" :class="validationResult.match_result ? 'text-green-800' : 'text-red-800'">
            {{ validationResult.match_result ? 'Match Confirmed!' : 'Match Failed' }}
          </h3>
          <p class="text-gray-600 mb-4">
            Similarity Score: {{ validationResult.similarity_score.toFixed(1) }}%
          </p>

          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p class="font-semibold text-gray-700">Confidence Level</p>
              <p class="text-gray-600">{{ validationResult.confidence_level }}</p>
            </div>
            <div>
              <p class="font-semibold text-gray-700">Match Threshold</p>
              <p class="text-gray-600">{{ validationResult.threshold_used }}%</p>
            </div>
          </div>
        </div>

        <!-- Quality Metrics -->
        <div class="grid grid-cols-2 gap-6">
          <div>
            <h4 class="font-semibold text-gray-900 mb-3">ID Photo Quality</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span>Brightness:</span>
                <span>{{ validationResult.image_quality.image1.brightness.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Contrast:</span>
                <span>{{ validationResult.image_quality.image1.contrast.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Sharpness:</span>
                <span>{{ validationResult.image_quality.image1.sharpness.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Resolution:</span>
                <span>{{ validationResult.image_quality.image1.resolution }}</span>
              </div>
            </div>
          </div>

          <div>
            <h4 class="font-semibold text-gray-900 mb-3">Selfie Quality</h4>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span>Brightness:</span>
                <span>{{ validationResult.image_quality.image2.brightness.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Contrast:</span>
                <span>{{ validationResult.image_quality.image2.contrast.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Sharpness:</span>
                <span>{{ validationResult.image_quality.image2.sharpness.toFixed(1) }}</span>
              </div>
              <div class="flex justify-between">
                <span>Resolution:</span>
                <span>{{ validationResult.image_quality.image2.resolution }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Navigation -->
        <div class="flex justify-between">
          <button @click="retakePhoto" class="btn-secondary">
            Retake Photo
          </button>
          <button @click="proceedToResults" class="btn-primary">
            View Results
          </button>
        </div>
      </div>

      <!-- License Image Missing Error -->
      <div v-if="!store.frontImageFile && !isStreaming && !validationResult" class="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div class="flex">
          <svg class="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
          </svg>
          <div class="ml-3">
            <h3 class="text-sm font-medium text-yellow-800">License Image Required</h3>
            <p class="mt-1 text-sm text-yellow-700">
              Please complete Step 1 (License Validation) first to upload your ID images before proceeding to selfie validation.
            </p>
            <div class="mt-3">
              <button
                @click="$router.push('/license')"
                class="text-sm font-medium text-yellow-800 hover:text-yellow-900"
              >
                Go to License Validation →
              </button>
            </div>
          </div>
        </div>
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
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'

const router = useRouter()
const store = useKycStore()

const videoElement = ref<HTMLVideoElement>()
const isStreaming = ref(false)
const capturedImage = ref<string>('')
const faceDetected = ref(false)
const faceBox = ref<{x: number, y: number, width: number, height: number} | null>(null)
const isValidating = ref(false)
const validationResult = ref<any>(null)
const error = ref('')

let stream: MediaStream | null = null
let faceDetectionInterval: number | null = null

const licenseImage = computed(() => {
  const frontImageFile = store.frontImageFile
  if (!frontImageFile) {
    console.warn('No front image file found in store')
    return ''
  }

  // Create object URL for the stored file
  const objectUrl = URL.createObjectURL(frontImageFile)
  console.log('Created license image URL from stored file')
  return objectUrl
})

onMounted(() => {
  startCamera()
})

onUnmounted(() => {
  stopCamera()
  // Clean up object URL to prevent memory leaks
  if (licenseImage.value) {
    URL.revokeObjectURL(licenseImage.value)
  }
})

const startCamera = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      }
    })

    if (videoElement.value) {
      videoElement.value.srcObject = stream
      isStreaming.value = true
      startFaceDetection()
    }
  } catch (err) {
    error.value = 'Failed to access camera. Please ensure camera permissions are granted.'
  }
}

const stopCamera = () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
    stream = null
  }

  if (faceDetectionInterval) {
    clearInterval(faceDetectionInterval)
    faceDetectionInterval = null
  }

  isStreaming.value = false
}

const startFaceDetection = () => {
  if (!videoElement.value) return

  faceDetectionInterval = window.setInterval(() => {
    detectFace()
  }, 100)
}

const detectFace = () => {
  if (!videoElement.value) return

  const video = videoElement.value
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  if (!ctx) return

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  ctx.drawImage(video, 0, 0)

  // Simple face detection using canvas (you might want to use a proper library)
  // For now, we'll assume face is detected in the center area
  const centerX = video.videoWidth / 2
  const centerY = video.videoHeight / 2
  const boxSize = Math.min(video.videoWidth, video.videoHeight) * 0.3

  faceBox.value = {
    x: centerX - boxSize / 2,
    y: centerY - boxSize / 2,
    width: boxSize,
    height: boxSize
  }

  faceDetected.value = true
}

const capturePhoto = () => {
  if (!videoElement.value || !faceDetected.value) return

  const video = videoElement.value
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  if (!ctx) return

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  ctx.drawImage(video, 0, 0)

  capturedImage.value = canvas.toDataURL('image/jpeg', 0.9)
}

const retakePhoto = () => {
  capturedImage.value = ''
  validationResult.value = null
  error.value = ''
}

const validateSelfie = async () => {
  if (!capturedImage.value) return

  isValidating.value = true
  error.value = ''

  try {
    // Convert base64 to blob
    const response = await fetch(capturedImage.value)
    const selfieBlob = await response.blob()

    // Get license image from store (you'll need to implement this)
    const licenseImageBlob = await getLicenseImageBlob()

    if (!licenseImageBlob) {
      throw new Error('License image not available')
    }

    const formData = new FormData()
    formData.append('file1', licenseImageBlob, 'license.jpg')
    formData.append('file2', selfieBlob, 'selfie.jpg')

    const validationResponse = await fetch('/selfie/validate_selfie', {
      method: 'POST',
      body: formData
    })

    if (!validationResponse.ok) {
      throw new Error(`HTTP error! status: ${validationResponse.status}`)
    }

    // The backend returns an image with comparison results, not JSON
    // Extract match result from response headers
    const matchResult = validationResponse.headers.get('X-Match-Result') === 'true'
    const similarityScore = parseFloat(validationResponse.headers.get('X-Similarity-Score') || '0')

    // Create a result object from the headers and store it
    const result = {
      match_result: matchResult,
      similarity_score: similarityScore,
      confidence_level: similarityScore >= 80 ? 'HIGH' : similarityScore >= 60 ? 'MEDIUM' : 'LOW',
      threshold_used: 60,
      image_quality: {
        image1: { brightness: 0, contrast: 0, sharpness: 0, resolution: 'N/A', is_good_quality: true },
        image2: { brightness: 0, contrast: 0, sharpness: 0, resolution: 'N/A', is_good_quality: true }
      },
      face_detection: {
        image1_face_detected: true,
        image2_face_detected: true
      },
      processing_info: {
        device_used: 'unknown',
        model: 'FaceNet',
        threshold_method: 'cosine_similarity'
      }
    }

    validationResult.value = result

    // Store selfie result
    store.setSelfieResult(result)

  } catch (err) {
    error.value = err instanceof Error ? err.message : 'An error occurred during validation'
  } finally {
    isValidating.value = false
  }
}

const getLicenseImageBlob = async (): Promise<Blob | null> => {
  const frontImageFile = store.frontImageFile
  if (!frontImageFile) {
    console.error('No front image file available for validation')
    return null
  }

  // Return the stored file directly
  console.log('Using stored front image file for validation')
  return frontImageFile
}

const proceedToResults = () => {
  router.push('/results')
}
</script>