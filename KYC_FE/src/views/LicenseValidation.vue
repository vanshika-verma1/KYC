<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import Cropper from 'cropperjs'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'
import {environments} from "@/env.ts"

const router = useRouter()
const store = useKycStore()

// Security check - ensure user has access to this route
onMounted(() => {
  // Check if user already completed license validation and should proceed to next step
  if (store.licenseResult) {
    // If license is already validated, redirect to next step (selfie)
    router.push('/selfie')
    return
  }
})

// Cleanup camera on component unmount
onUnmounted(() => {
  if (cameraStream.value) {
    cameraStream.value.getTracks().forEach(track => track.stop())
  }
})

// File input refs
const frontInput = ref<HTMLInputElement>()
const backInput = ref<HTMLInputElement>()

// Camera refs and state
const cameraModal = ref(false)
const cameraStream = ref<MediaStream | null>(null)
const videoElement = ref<HTMLVideoElement>()
const canvasElement = ref<HTMLCanvasElement>()
const currentSide = ref<'front' | 'back'>('front')
const cameraError = ref('')
const videoLoading = ref(false)

// Cropping refs and state
const cropModal = ref(false)
const cropImageElement = ref<HTMLImageElement>()
const cropSide = ref<'front' | 'back'>('front')
const cropperInstance = ref<Cropper | null>(null)

// Image and file state
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

// Camera functions
const startCamera = async (side: 'front' | 'back') => {
  try {
    cameraError.value = ''
    currentSide.value = side
    videoLoading.value = true

    const constraints: MediaStreamConstraints = {
      video: {
        facingMode: 'environment', // Use back camera for better quality
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    cameraStream.value = stream
    cameraModal.value = true

    // Wait for next DOM update to ensure video element is available
    await nextTick()
    if (videoElement.value) {
      videoElement.value.srcObject = stream
      // Add event listener for when video is ready
      videoElement.value.onloadedmetadata = () => {
        videoLoading.value = false
      }
    }
  } catch (err) {
    cameraError.value = err instanceof Error ? err.message : 'Failed to access camera'
    console.error('Camera access error:', err)
    videoLoading.value = false
  }
}

const stopCamera = () => {
  if (cameraStream.value) {
    cameraStream.value.getTracks().forEach(track => track.stop())
    cameraStream.value = null
  }
  cameraModal.value = false
}

const capturePhoto = () => {
  try {
    // Check if all required elements are available
    if (!videoElement.value) {
      cameraError.value = 'Video element not available'
      return
    }

    if (!canvasElement.value) {
      cameraError.value = 'Canvas element not available'
      return
    }

    if (!cameraStream.value) {
      cameraError.value = 'Camera stream not available'
      return
    }

    const video = videoElement.value
    const canvas = canvasElement.value

    // Ensure video is ready and has dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      cameraError.value = 'Video not ready. Please wait a moment and try again.'
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      cameraError.value = 'Could not get canvas context'
      return
    }

    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0)

    // Convert canvas to blob with error handling
    canvas.toBlob((blob) => {
      if (!blob) {
        cameraError.value = 'Failed to capture image. Please try again.'
        return
      }

      try {
        // Create a File object from the blob
        const timestamp = new Date().getTime()
        const fileName = `${currentSide.value}_${timestamp}.jpg`
        const file = new File([blob], fileName, { type: 'image/jpeg' })

        // Update the appropriate image state
        if (currentSide.value === 'front') {
          frontFile.value = file
          const reader = new FileReader()
          reader.onload = (e) => {
            frontImage.value = e.target?.result as string
          }
          reader.onerror = () => {
            cameraError.value = 'Failed to process front image'
          }
          reader.readAsDataURL(file)
        } else {
          backFile.value = file
          const reader = new FileReader()
          reader.onload = (e) => {
            backImage.value = e.target?.result as string
          }
          reader.onerror = () => {
            cameraError.value = 'Failed to process back image'
          }
          reader.readAsDataURL(file)
        }

        // Stop camera and close modal after successful capture
        stopCamera()
      } catch (err) {
        cameraError.value = 'Failed to create image file'
        console.error('File creation error:', err)
      }
    }, 'image/jpeg', 0.9)
  } catch (err) {
    cameraError.value = 'An unexpected error occurred during capture'
    console.error('Capture error:', err)
  }
}

const takePhoto = (side: 'front' | 'back') => {
  startCamera(side)
}

const closeCamera = () => {
  stopCamera()
}

// Cropping functions using Cropper.js
const openCropModal = (side: 'front' | 'back') => {
  cropSide.value = side
  cropModal.value = true

  // Initialize Cropper.js after modal opens
  nextTick(() => {
    if (cropImageElement.value) {
      // Destroy existing cropper if it exists
      if (cropperInstance.value) {
        if (typeof (cropperInstance.value as any).cropper !== 'undefined') {
          (cropperInstance.value as any).cropper.destroy()
        }
      }

      // Create new cropper instance with basic options
      cropperInstance.value = new Cropper(cropImageElement.value)

      console.log('Cropper initialized for', side)
    }
  })
}

const closeCropModal = () => {
  cropModal.value = false

  // Clean up cropper instance
  if (cropperInstance.value) {
    // Use the correct cleanup method
    if (typeof (cropperInstance.value as any).cropper !== 'undefined') {
      (cropperInstance.value as any).cropper.destroy()
    }
    cropperInstance.value = null
  }
}

const applyCrop = async () => {
  if (!cropperInstance.value) {
    console.error('Cropper instance not available')
    return
  }

  try {
    // Get cropped canvas using the correct method
    const canvas = (cropperInstance.value as any).getCropperCanvas()

    if (!canvas) {
      console.error('Failed to get cropped canvas')
      return
    }

    // Convert canvas to blob
    canvas.toBlob((blob: Blob | null) => {
      if (!blob) {
        console.error('Failed to create blob from cropped canvas')
        return
      }

      try {
        const timestamp = new Date().getTime()
        const fileName = `cropped_${cropSide.value}_${timestamp}.jpg`
        const file = new File([blob], fileName, { type: 'image/jpeg' })

        // Update the appropriate image state
        if (cropSide.value === 'front') {
          frontFile.value = file
          const reader = new FileReader()
          reader.onload = (e) => {
            if (e.target?.result) {
              frontImage.value = e.target.result as string
            }
          }
          reader.onerror = (error) => {
            console.error('FileReader error:', error)
          }
          reader.readAsDataURL(file)
        } else {
          backFile.value = file
          const reader = new FileReader()
          reader.onload = (e) => {
            if (e.target?.result) {
              backImage.value = e.target.result as string
            }
          }
          reader.onerror = (error) => {
            console.error('FileReader error:', error)
          }
          reader.readAsDataURL(file)
        }

        closeCropModal()
      } catch (error) {
        console.error('Error creating cropped file:', error)
      }
    }, 'image/jpeg', 0.9)
  } catch (error) {
    console.error('Error during crop application:', error)
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

    const response = await fetch(`${environments.baseurl}/license/validate_license`, {
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

    // Update verification state - mark license step as completed
    if ((window as any).verificationState) {
      (window as any).verificationState.completeStep('license')
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

<template>
  <div class="max-w-4xl mx-auto">
    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6">
      <div class="text-center mb-6">
        <div class="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h2 class="text-2xl font-semibold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-2">
          License Validation
        </h2>
        <p class="text-gray-600 text-md">
          Please upload clear photos of both sides of your government-issued License ID.
        </p>
      </div>

      <!-- Upload Sections -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6 items-start">

        <!-- Front Image Upload -->
        <div class="space-y-3 h-full">
          <div class="text-center">
            <h3 class="text-xl font-semibold text-gray-800 mb-1">Front Side</h3>
            <p class="text-sm text-gray-500">Upload or take a photo of the front of your ID card</p>
          </div>

          <!-- Upload/Preview Section -->
          <div class="relative group">
            <div :class="[
              'border-2 border-dashed rounded-2xl p-5 text-center transition-all duration-300 cursor-pointer min-h-[280px] flex flex-col justify-center',
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
                <div class="mx-auto w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg class="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div class="flex flex-col space-y-2">
                  <button class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all">
                    Choose Front Image
                  </button>
                  <div class="text-xs text-gray-500">or</div>
                  <button @click.stop="takePhoto('front')" class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all">
                    <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    Take Photo
                  </button>
                </div>
              </div>
              <div v-else class="space-y-2 flex-1 flex flex-col justify-center">
                <div class="relative">
                  <img :src="frontImage" alt="Front side" class="w-full h-36 object-contain rounded-xl" />
                  <div class="absolute top-2 right-2">
                    <div class="bg-green-500 text-white text-xs font-medium px-2 py-1 rounded-full">
                      ✓ Ready
                    </div>
                  </div>
                </div>
                <div class="flex justify-center space-x-4">
                  <button @click.stop="openCropModal('front')" class="bg-blue-50 hover:bg-blue-100 text-blue-600 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-blue-150">
                    <svg class="w-4 h-4 mr-1 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                    Crop
                  </button>
                  <button @click.stop="removeFrontImage" class="bg-red-50 hover:bg-red-100 text-red-600 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-red-150">
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Back Image Upload -->
        <div class="space-y-3 h-full">
          <div class="text-center">
            <h3 class="text-xl font-semibold text-gray-800 mb-1">Back Side</h3>
            <p class="text-sm text-gray-500">Upload or take a photo of the back of your ID card</p>
          </div>

          <!-- Upload/Preview Section -->
          <div class="relative group">
            <div :class="[
              'border-2 border-dashed rounded-2xl p-5 text-center transition-all duration-300 cursor-pointer min-h-[280px] flex flex-col justify-center',
              'hover:scale-[1.02] hover:shadow-lg',
              backImage ? 'border-green-300 bg-green-50/50' : 'border-gray-300 hover:border-purple-400 bg-gray-50/50'
            ]" @click="triggerBackInput">
              <input
                ref="backInput"
                type="file"
                accept="image/*"
                @change="onBackImageChange"
                class="hidden"
              />
              <div v-if="!backImage" class="space-y-4">
                <div class="mx-auto w-16 h-16 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                  <svg class="w-8 h-8 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div class="flex flex-col space-y-2">
                  <button class="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all">
                    Choose Back Image
                  </button>
                  <div class="text-xs text-gray-500">or</div>
                  <button @click.stop="takePhoto('back')" class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all">
                    <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    Take Photo
                  </button>
                </div>
              </div>
              <div v-else class="space-y-2 flex-1 flex flex-col justify-center">
                <div class="relative">
                  <img :src="backImage" alt="Back side" class="w-full h-36 object-contain rounded-xl" />
                  <div class="absolute top-2 right-2">
                    <div class="bg-green-500 text-white text-xs font-medium px-2 py-1 rounded-full">
                      ✓ Ready
                    </div>
                  </div>
                </div>
                <div class="flex justify-center space-x-4">
                  <button @click.stop="openCropModal('back')" class="bg-blue-50 hover:bg-blue-100 text-blue-600 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-blue-150">
                    <svg class="w-4 h-4 mr-1 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                    Crop
                  </button>
                  <button @click.stop="removeBackImage" class="bg-red-50 hover:bg-red-100 text-red-600 text-sm font-medium py-2 px-4 rounded-lg transition-colors border-1 border-red-150">
                    Remove
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Submit Button -->
      <div class="text-center pt-2">
        <button
          @click="validateLicense"
          :disabled="!frontImage || !backImage || isLoading"
          :class="[
            'relative px-8 py-3 rounded-xl font-medium text-base transition-all duration-300 transform',
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
          <div class="flex items-center space-x-2 text-xs">
            <div :class="frontImage ? 'w-3 h-3 bg-green-500 rounded-full' : 'w-3 h-3 bg-gray-300 rounded-full'"></div>
            <span :class="frontImage ? 'text-green-600 font-medium' : 'text-gray-500'">Front Image</span>
          </div>
          <div class="flex items-center space-x-2 text-xs">
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

  <!-- Hidden canvas for photo capture -->
  <canvas ref="canvasElement" class="hidden"></canvas>

  <!-- Hidden canvas for cropping -->
  <canvas ref="cropCanvasElement" class="hidden"></canvas>

  <!-- Camera Modal -->
  <div v-if="cameraModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-2xl shadow-2xl w-full max-w-md">
      <!-- Header -->
      <div class="flex justify-between items-center p-4 border-b">
        <h3 class="text-md font-semibold text-gray-800">
          Capture {{ currentSide === 'front' ? 'front' : 'back' }} side of the License
        </h3>
        <button @click="closeCamera" class="text-gray-500 hover:text-gray-700 p-1">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Camera View -->
      <div class="relative bg-black aspect-[4/3]">
        <video
          ref="videoElement"
          autoplay
          playsinline
          muted
          class="w-full h-full object-cover"
        ></video>

        <!-- Loading overlay -->
        <div v-if="videoLoading" class="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div class="text-white text-center">
            <svg class="animate-spin h-8 w-8 mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p class="text-sm">Starting camera...</p>
          </div>
        </div>

        <!-- Camera overlay for better composition -->
        <div class="absolute inset-0 border-2 border-white rounded-lg m-4 pointer-events-none">
          <div class="w-full h-full border-2 border-white rounded-lg"></div>
        </div>
      </div>

      <!-- Controls -->
      <div class="p-4 flex justify-center space-x-4 text-sm">
        <button @click="closeCamera" class="px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg font-medium transition-colors">
          Cancel
        </button>
        <button
          @click="capturePhoto"
          :disabled="videoLoading || !!cameraError"
          :class="[
            'px-6 py-2 rounded-lg font-medium transition-all shadow-lg hover:shadow-xl',
            (videoLoading || !!cameraError)
              ? 'bg-gray-400 cursor-not-allowed text-gray-600'
              : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white'
          ]"
        >
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          {{ videoLoading ? 'Loading...' : 'Capture' }}
        </button>
      </div>

      <!-- Error Message -->
      <div v-if="cameraError" class="p-4 bg-red-50 border-t">
        <p class="text-sm text-red-600 text-center">{{ cameraError }}</p>
      </div>
    </div>
  </div>

  <!-- Cropping Modal -->
  <div v-if="cropModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-2xl shadow-2xl w-full max-w-4xl">
      <!-- Header -->
      <div class="flex justify-between items-center p-4 border-b">
        <h3 class="text-lg font-semibold text-gray-800">
          Crop {{ cropSide === 'front' ? 'Front' : 'Back' }} License Image
        </h3>
        <button @click="closeCropModal" class="text-gray-500 hover:text-gray-700 p-1">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Cropping Area -->
      <div class="p-4">
        <div class="bg-gray-100 rounded-lg overflow-hidden max-h-96">
          <img
            :src="cropSide === 'front' ? frontImage : backImage"
            ref="cropImageElement"
            alt="Crop area"
            class="cropper-image w-full h-full object-contain"
          />
        </div>

        <!-- Instructions -->
        <div class="mt-4 text-center text-sm text-gray-600">
          <p>Drag to adjust the crop area, then click "Apply Crop"</p>
        </div>
      </div>

      <!-- Controls -->
      <div class="p-4 border-t flex justify-center space-x-4">
        <button @click="closeCropModal" class="px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg font-medium transition-colors">
          Cancel
        </button>
        <button @click="applyCrop" class="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white rounded-lg font-medium transition-all shadow-lg hover:shadow-xl">
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          Apply Crop
        </button>
      </div>
    </div>
  </div>
</template>