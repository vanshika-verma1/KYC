<template>
  <div class="max-w-6xl mx-auto">
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
          Please take a clear photo of your face. We'll compare it with your ID photo to verify your identity.
        </p>
      </div>

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

      <!-- Image Preview/Camera Section -->
      <div class="mb-8">
        <!-- Image Preview Mode -->
        <div v-if="capturedImage || uploadedImage" class="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl overflow-hidden aspect-video shadow-2xl">
          <img
            :src="capturedImage || uploadedImage"
            alt="Selfie preview"
            class="w-full h-full object-cover"
          />
          <!-- Image selected overlay -->
          <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div class="text-center text-white">
              <div class="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p class="text-xl font-semibold">{{ capturedImage ? 'Photo Captured!' : 'Image Uploaded!' }}</p>
              <p class="text-sm opacity-75">Ready for validation</p>
            </div>
          </div>
        </div>

        <!-- Camera Mode -->
        <div v-else class="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl overflow-hidden aspect-video shadow-2xl">
          <video
            v-if="isStreaming"
            ref="videoElement"
            autoplay
            playsinline
            muted
            class="w-full h-full object-cover"
          ></video>

          <!-- Camera overlay when not streaming -->
          <div v-if="!isStreaming" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40">
            <div class="text-center text-white">
              <div class="w-20 h-20 bg-white bg-opacity-20 rounded-full mx-auto mb-6 flex items-center justify-center">
                <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <p class="text-xl font-semibold mb-2">Choose how to provide your selfie</p>
              <p class="text-sm opacity-75">Upload an existing photo or take a new one</p>
            </div>
          </div>

          <!-- Camera streaming overlay -->
          <div v-if="isStreaming && !capturedImage" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40">
            <div class="text-center text-white">
              <div class="animate-pulse w-20 h-20 bg-white bg-opacity-20 rounded-full mx-auto mb-6 flex items-center justify-center">
                <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <p class="text-xl font-semibold mb-2">Position your face in the center</p>
              <p class="text-sm opacity-75">Ensure good lighting and look directly at the camera</p>
            </div>
          </div>

          <!-- Enhanced face detection box -->
          <div
            v-if="faceBox && isStreaming && !capturedImage"
            class="absolute border-2 border-green-400 rounded-lg shadow-lg shadow-green-400/30 animate-pulse"
            :style="{
              left: `${faceBox.x}px`,
              top: `${faceBox.y}px`,
              width: `${faceBox.width}px`,
              height: `${faceBox.height}px`
            }"
          ></div>
        </div>
      </div>

      <!-- Enhanced Controls -->
      <div class="mb-8">
        <!-- Initial Options - Upload or Capture -->
        <div v-if="!capturedImage && !uploadedImage" class="flex flex-wrap justify-center gap-4 mb-8">
          <!-- Upload Option -->
          <div class="relative">
            <input
              ref="selfieInput"
              type="file"
              accept="image/*"
              @change="onSelfieImageChange"
              class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-semibold py-3 px-8 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105">
              <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Upload Image
            </button>
          </div>

          <button
            @click="startCamera"
            class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-8 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
          >
            <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Take Photo
          </button>
        </div>

        <!-- Camera Controls -->
        <div v-if="isStreaming && !capturedImage && !uploadedImage" class="flex justify-center">
          <button
            @click="capturePhoto"
            :disabled="!faceDetected"
            :class="[
              'font-semibold py-3 px-8 rounded-xl shadow-lg transition-all transform',
              faceDetected && !isValidating
                ? 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white hover:shadow-xl hover:scale-105'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            ]"
          >
            <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            {{ faceDetected ? 'Capture Photo' : 'Detecting Face...' }}
          </button>
        </div>

        <!-- Validation Button - appears after image is selected -->
        <div v-if="(capturedImage || uploadedImage) && !isValidating" class="text-center">
          <button
            @click="validateSelfie"
            class="bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white font-semibold py-4 px-12 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105 text-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            <svg class="w-6 h-6 mr-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Validate & Continue
          </button>

          <!-- Image info -->
          <div class="mt-3 text-sm text-gray-500">
            <span v-if="capturedImage">📸 Camera photo ready for validation</span>
            <span v-if="uploadedImage">📁 Uploaded image ready for validation</span>
          </div>

          <!-- Retake option -->
          <div class="mt-4 space-x-4">
            <button
              v-if="capturedImage"
              @click="retakePhoto"
              class="text-sm text-gray-500 hover:text-gray-700 underline transition-colors"
            >
              Or retake photo
            </button>
            <button
              v-if="uploadedImage"
              @click="clearUploadedImage"
              class="text-sm text-gray-500 hover:text-gray-700 underline transition-colors"
            >
              Or choose different image
            </button>
          </div>
        </div>

        <!-- Stop Camera button -->
        <div v-if="isStreaming && !capturedImage && !uploadedImage" class="text-center mt-4">
          <button
            @click="stopCamera"
            class="text-sm text-gray-500 hover:text-gray-700 underline"
          >
            Cancel & Close Camera
          </button>
        </div>
      </div>

      <!-- Enhanced Loading State -->
      <div v-if="isValidating" class="text-center py-12">
        <div class="relative mx-auto mb-8">
          <div class="animate-spin w-20 h-20 border-4 border-purple-200 border-t-purple-600 rounded-full"></div>
          <div class="absolute inset-0 flex items-center justify-center">
            <svg class="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
        </div>
        <p class="text-xl font-semibold text-gray-700 mb-2">Comparing faces...</p>
        <p class="text-sm text-gray-500 mb-4">Analyzing facial features and verifying identity</p>

        <!-- Progress steps -->
        <div class="flex justify-center items-center space-x-4 text-sm text-gray-600">
          <div class="flex items-center space-x-2">
            <div class="w-2 h-2 bg-purple-400 rounded-full"></div>
            <span>Face Detection</span>
          </div>
          <div class="w-6 h-px bg-purple-300"></div>
          <div class="flex items-center space-x-2">
            <div class="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
            <span>Feature Extraction</span>
          </div>
          <div class="w-6 h-px bg-purple-300"></div>
          <div class="flex items-center space-x-2">
            <div class="w-2 h-2 bg-purple-300 rounded-full"></div>
            <span>Comparison</span>
          </div>
        </div>

        <div class="mt-6 flex justify-center space-x-1">
          <div class="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
          <div class="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style="animation-delay: 0.2s"></div>
          <div class="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style="animation-delay: 0.4s"></div>
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

      <!-- Enhanced Error Message -->
      <div v-if="error" class="mt-8 p-6 bg-red-50/80 backdrop-blur-sm border border-red-200 rounded-2xl">
        <div class="flex items-start">
          <div class="flex-shrink-0">
            <svg class="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="ml-3 flex-1">
            <h3 class="text-sm font-semibold text-red-800">Validation Error</h3>
            <p class="mt-1 text-sm text-red-700">{{ error }}</p>

            <!-- Helpful suggestions based on error type -->
            <div v-if="error.includes('license')" class="mt-3 p-3 bg-red-100/50 rounded-lg">
              <p class="text-sm font-medium text-red-800 mb-1">To fix this issue:</p>
              <ol class="text-sm text-red-700 list-decimal list-inside space-y-1">
                <li>Go back to License Validation step</li>
                <li>Upload a clear photo of your ID document</li>
                <li>Ensure all corners of the document are visible</li>
                <li>Try again once the license validation is complete</li>
              </ol>
            </div>

            <div v-if="error.includes('face') || error.includes('No face detected')" class="mt-3 p-3 bg-red-100/50 rounded-lg">
              <p class="text-sm font-medium text-red-800 mb-1">To fix this issue:</p>
              <ol class="text-sm text-red-700 list-decimal list-inside space-y-1">
                <li>Ensure your face is clearly visible in the photo</li>
                <li>Remove sunglasses, hats, or other face coverings</li>
                <li>Ensure good lighting on your face</li>
                <li>Look directly at the camera</li>
                <li>Try taking the photo again</li>
              </ol>
            </div>

            <div v-if="error.includes('network') || error.includes('fetch')" class="mt-3 p-3 bg-red-100/50 rounded-lg">
              <p class="text-sm font-medium text-red-800 mb-1">Network Error:</p>
              <ol class="text-sm text-red-700 list-decimal list-inside space-y-1">
                <li>Check your internet connection</li>
                <li>Ensure the backend server is running on port 8000</li>
                <li>Try again in a few moments</li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Retry button for failed validations -->
    <div v-if="error && (capturedImage || uploadedImage)" class="mt-4">
      <button
        @click="retryValidation"
        class="text-sm bg-purple-100 hover:bg-purple-200 text-purple-700 px-4 py-2 rounded-lg transition-colors"
      >
        <svg class="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        Try Again
      </button>
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
const selfieInput = ref<HTMLInputElement>()
const isStreaming = ref(false)
const capturedImage = ref<string>('')
const uploadedImage = ref<string>('')
const faceDetected = ref(false)
const faceBox = ref<{x: number, y: number, width: number, height: number} | null>(null)
const isValidating = ref(false)
const validationResult = ref<any>(null)
const comparisonImage = ref<string>('')
const error = ref('')

let stream: MediaStream | null = null
let faceDetectionInterval: number | null = null

const licenseImage = computed(() => {
  const frontImageFile = store.frontImageFile
  if (!frontImageFile) {
    console.warn('No front image file found in store')
    return ''
  }

  try {
    // Create object URL for the stored file
    const objectUrl = URL.createObjectURL(frontImageFile)
    console.log('Created license image URL from stored file')
    return objectUrl
  } catch (error) {
    console.error('Error creating object URL for license image:', error)
    return ''
  }
})

onMounted(() => {
  // Camera no longer auto-starts - user chooses upload or camera
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

  // Check if video is ready and has dimensions
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    faceDetected.value = false
    return
  }

  // Simple face detection using canvas and basic image processing
  // Look for areas with higher contrast (likely to be faces)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  if (!ctx) return

  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  ctx.drawImage(video, 0, 0)

  // Get image data for analysis
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = imageData.data

  // Simple face detection by finding areas with skin-like colors and contrast
  const centerX = canvas.width / 2
  const centerY = canvas.height / 2

  // Assume face is in the central area for now (you can implement real face detection later)
  const boxSize = Math.min(canvas.width, canvas.height) * 0.4

  // Check if there's enough contrast in the center area (indicating a face might be there)
  let hasContrast = false
  const sampleSize = 50

  for (let x = centerX - sampleSize; x < centerX + sampleSize; x += 10) {
    for (let y = centerY - sampleSize; y < centerY + sampleSize; y += 10) {
      if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
        const i = (y * canvas.width + x) * 4
        const r = data[i]
        const g = data[i + 1]
        const b = data[i + 2]

        // Simple skin detection (basic heuristic)
        if (r > 60 && g > 40 && b > 20 && r > g && r > b) {
          hasContrast = true
          break
        }
      }
    }
    if (hasContrast) break
  }

  if (hasContrast) {
    faceBox.value = {
      x: centerX - boxSize / 2,
      y: centerY - boxSize / 2,
      width: boxSize,
      height: boxSize
    }
    faceDetected.value = true
  } else {
    faceDetected.value = false
    faceBox.value = null
  }
}

const capturePhoto = async () => {
   if (!videoElement.value || !faceDetected.value) {
     error.value = 'Please wait for face detection before capturing'
     return
   }

   try {
     const video = videoElement.value
     const canvas = document.createElement('canvas')
     const ctx = canvas.getContext('2d')

     if (!ctx) {
       error.value = 'Unable to capture image - canvas not supported'
       return
     }

     // Set canvas dimensions to match video
     canvas.width = video.videoWidth
     canvas.height = video.videoHeight

     // Draw the current video frame
     ctx.drawImage(video, 0, 0)

     // Convert to base64 with good quality
     const imageData = canvas.toDataURL('image/jpeg', 0.9)

     if (imageData && imageData.length > 100) { // Basic check for valid image
       capturedImage.value = imageData

       // Stop camera after capturing
       stopCamera()

       // Clear uploaded image if exists
       uploadedImage.value = ''

       // Clear any previous errors
       error.value = ''

       console.log('Photo captured successfully:', {
         width: video.videoWidth,
         height: video.videoHeight,
         size: imageData.length
       })
     } else {
       error.value = 'Failed to capture image - please try again'
     }
   } catch (err) {
     console.error('Error capturing photo:', err)
     error.value = 'Failed to capture photo. Please try again.'
   }
 }

const retakePhoto = () => {
   capturedImage.value = ''
   uploadedImage.value = ''
   validationResult.value = null
   error.value = ''
 }

const clearUploadedImage = () => {
   uploadedImage.value = ''
   validationResult.value = null
   error.value = ''
 }


const validateSelfie = async () => {
  // Use either captured or uploaded image
  const imageToValidate = capturedImage.value || uploadedImage.value
  if (!imageToValidate) {
    error.value = 'Please capture or upload a selfie image first'
    return
  }

  // Validate license image is available
  if (!store.frontImageFile) {
    error.value = 'License image not available. Please complete license validation first.'
    return
  }

  isValidating.value = true
  error.value = ''

  try {
    // Convert base64 to blob for selfie
    const selfieResponse = await fetch(imageToValidate)
    const selfieBlob = await selfieResponse.blob()

    const formData = new FormData()
    // Backend expects file1 as license image and file2 as selfie
    formData.append('file1', store.frontImageFile, 'license.jpg')
    formData.append('file2', selfieBlob, 'selfie.jpg')

    console.log('Sending validation request with files:', {
      licenseSize: store.frontImageFile.size,
      selfieSize: selfieBlob.size
    })

    // Use the correct API endpoint based on backend routing
    const validationResponse = await fetch('http://localhost:8000/selfie/validate_selfie', {
      method: 'POST',
      body: formData
    })

    if (!validationResponse.ok) {
      const errorText = await validationResponse.text().catch(() => 'Unknown error')
      throw new Error(`Validation failed: ${validationResponse.status} - ${errorText}`)
    }

    // Get match result from response headers
    const matchResult = validationResponse.headers.get('X-Match-Result') === 'true'
    const similarityScore = parseFloat(validationResponse.headers.get('X-Similarity-Score') || '0')

    if (isNaN(similarityScore)) {
      throw new Error('Invalid response from server - no similarity score received')
    }

    console.log('Validation successful:', { matchResult, similarityScore })

    // Create a comprehensive result object
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
        device_used: 'web_browser',
        model: 'FaceNet',
        threshold_method: 'cosine_similarity'
      }
    }

    validationResult.value = result

    // Store selfie result in the store
    store.setSelfieResult(result)

    // Navigate to results page
    router.push('/results')

  } catch (err) {
    console.error('Selfie validation error:', err)
    error.value = err instanceof Error ? err.message : 'An error occurred during validation'
    isValidating.value = false
  }
}

// License image is now handled directly from store

const onSelfieImageChange = (event: Event) => {
   const target = event.target as HTMLInputElement
   const file = target.files?.[0]
   if (file) {
     // Validate file type
     if (!file.type.startsWith('image/')) {
       error.value = 'Please select a valid image file'
       return
     }

     // Validate file size (max 10MB)
     if (file.size > 10 * 1024 * 1024) {
       error.value = 'Image file size must be less than 10MB'
       return
     }

     const reader = new FileReader()
     reader.onload = (e) => {
       const result = e.target?.result as string
       if (result) {
         uploadedImage.value = result
         // Stop camera if it's running when user uploads an image
         if (isStreaming.value) {
           stopCamera()
         }
         // Clear captured image if exists
         capturedImage.value = ''
         // Clear any previous errors
         error.value = ''
         console.log('Image uploaded successfully:', {
           name: file.name,
           size: file.size,
           type: file.type
         })
       }
     }
     reader.onerror = () => {
       error.value = 'Failed to read the uploaded image file'
     }
     reader.readAsDataURL(file)
   }
 }

const retryValidation = () => {
  error.value = ''
  isValidating.value = false
  validationResult.value = null
  // The existing image will still be there, so the user can try again
}

const proceedToResults = () => {
  router.push('/results')
}
</script>