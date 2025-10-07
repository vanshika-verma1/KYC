<template>
  <div class="max-w-4xl mx-auto">
    <div class="card">
      <h2 class="text-3xl font-bold text-gray-900 mb-2">Liveness Detection</h2>
      <p class="text-gray-600 mb-8">
        Please follow the instructions to verify you're a real person. The system will guide you through the process.
      </p>

      <!-- Video Stream -->
      <div class="mb-8">
        <div class="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
          <video
            ref="videoElement"
            autoplay
            playsinline
            muted
            class="w-full h-full object-cover"
          ></video>

          <!-- Overlay for instructions -->
          <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div class="text-center text-white">
              <div v-if="isConnecting" class="space-y-4">
                <div class="animate-spin w-12 h-12 border-4 border-white border-t-transparent rounded-full mx-auto"></div>
                <p class="text-lg">Connecting to liveness detection...</p>
              </div>

              <div v-else-if="currentInstruction" class="space-y-4">
                <div class="text-6xl">
                  {{ getInstructionIcon(currentInstruction) }}
                </div>
                <p class="text-xl font-semibold">{{ getInstructionText(currentInstruction) }}</p>
                <p class="text-sm opacity-75">{{ getInstructionDetail(currentInstruction) }}</p>
              </div>

              <div v-else class="space-y-4">
                <div class="animate-pulse w-12 h-12 bg-white bg-opacity-25 rounded-full mx-auto"></div>
                <p class="text-lg">Initializing camera...</p>
              </div>
            </div>
          </div>

          <!-- Status indicators -->
          <div class="absolute top-4 left-4 space-y-2">
            <div class="flex items-center space-x-2 bg-black bg-opacity-50 px-3 py-1 rounded-full">
              <div
                :class="[
                  'w-3 h-3 rounded-full',
                  wsConnected ? 'bg-green-400' : 'bg-red-400'
                ]"
              ></div>
              <span class="text-white text-sm">
                {{ wsConnected ? 'Connected' : 'Disconnected' }}
              </span>
            </div>

            <div class="flex items-center space-x-2 bg-black bg-opacity-50 px-3 py-1 rounded-full">
              <div
                :class="[
                  'w-3 h-3 rounded-full',
                  livenessStatus.blinks >= 2 ? 'bg-green-400' : 'bg-yellow-400'
                ]"
              ></div>
              <span class="text-white text-sm">
                Blinks: {{ livenessStatus.blinks }}/2
              </span>
            </div>

            <div class="flex items-center space-x-2 bg-black bg-opacity-50 px-3 py-1 rounded-full">
              <div
                :class="[
                  'w-3 h-3 rounded-full',
                  Math.abs(livenessStatus.yaw) > 20 ? 'bg-green-400' : 'bg-yellow-400'
                ]"
              ></div>
              <span class="text-white text-sm">
                Head Turn: {{ Math.abs(livenessStatus.yaw).toFixed(1) }}°
              </span>
            </div>
          </div>

          <!-- Bounding box overlay -->
          <div
            v-if="livenessStatus.bbox"
            class="absolute border-2 border-blue-500"
            :style="{
              left: `${livenessStatus.bbox[0]}px`,
              top: `${livenessStatus.bbox[1]}px`,
              width: `${livenessStatus.bbox[2] - livenessStatus.bbox[0]}px`,
              height: `${livenessStatus.bbox[3] - livenessStatus.bbox[1]}px`
            }"
          ></div>
        </div>
      </div>

      <!-- Controls -->
      <div class="flex justify-between items-center">
        <button
          @click="startCamera"
          :disabled="isStreaming"
          class="btn-secondary"
        >
          {{ isStreaming ? 'Camera Active' : 'Start Camera' }}
        </button>

        <div class="flex space-x-4">
          <button
            @click="restartProcess"
            class="btn-secondary"
          >
            Restart
          </button>

          <button
            @click="proceedToSelfie"
            :disabled="!isLivenessComplete"
            :class="[
              'px-8 py-3 rounded-lg font-medium transition-all',
              isLivenessComplete
                ? 'bg-green-600 hover:bg-green-700 text-white animate-pulse'
                : 'bg-primary-500 hover:bg-primary-600 text-white opacity-50 cursor-not-allowed'
            ]"
          >
            {{ isLivenessComplete ? '✅ Continue to Selfie' : 'Continue to Selfie' }}
          </button>
        </div>
      </div>

      <!-- Progress indicators -->
      <div class="mt-8 grid grid-cols-3 gap-4">
        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl mb-2">
            {{ livenessStatus.blinks >= 2 ? '✅' : '👁️' }}
          </div>
          <p class="font-semibold">Blink Detection</p>
          <p class="text-sm text-gray-600">{{ livenessStatus.blinks }}/2 blinks</p>
        </div>

        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl mb-2">
            {{ Math.abs(livenessStatus.yaw || 0) > 20 ? '✅' : '↺' }}
          </div>
          <p class="font-semibold">Head Movement</p>
          <p class="text-sm text-gray-600">{{ Math.abs(livenessStatus.yaw || 0).toFixed(1) }}° rotation</p>
        </div>

        <div class="text-center p-4 bg-gray-50 rounded-lg">
          <div class="text-2xl mb-2">
            {{ isSpoofCheckPassed ? '✅' : spoofBuffer.length >= MIN_SPOOF_BUFFER_SIZE ? '⚠️' : '⏳' }}
          </div>
          <p class="font-semibold">Liveness Check</p>
          <p class="text-sm text-gray-600">
            <span v-if="isSpoofCheckPassed">Live person</span>
            <span v-else-if="spoofBuffer.length >= MIN_SPOOF_BUFFER_SIZE">Spoof risk</span>
            <span v-else>Analyzing...</span>
          </p>
          <p class="text-xs text-gray-500">
            Buffer: {{ spoofBuffer.length }}/3
          </p>
        </div>
      </div>

      <!-- Debug Info (only in development) -->
      <div class="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <details class="text-sm">
          <summary class="font-semibold text-blue-800 cursor-pointer">Debug Info (Click to expand)</summary>
          <div class="mt-2 space-y-1 text-blue-700">
            <p>Raw blinks: {{ livenessStatus.blinks }}</p>
            <p>Raw yaw: {{ livenessStatus.yaw }}</p>
            <p>Raw spoof_prob: {{ livenessStatus.spoof_prob }}</p>
            <p>Abs yaw: {{ Math.abs(livenessStatus.yaw || 0) }}</p>
            <p>Spoof buffer: [{{ spoofBuffer.map(v => v?.toFixed(2)).join(', ') }}]</p>
            <p>Buffer size: {{ spoofBuffer.length }}/{{ MIN_SPOOF_BUFFER_SIZE }}</p>
            <p>Blinks >= 2: {{ (livenessStatus.blinks || 0) >= 2 ? '✅' : '❌' }}</p>
            <p>Yaw > 20: {{ Math.abs(livenessStatus.yaw || 0) > 20 ? '✅' : '❌' }}</p>
            <p>Spoof avg < 0.7: {{ isSpoofCheckPassed ? '✅' : '❌' }}</p>
            <p><strong>Overall Complete: {{ isLivenessComplete ? '✅ YES' : '❌ NO' }}</strong></p>
          </div>
        </details>
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
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'

const router = useRouter()
const store = useKycStore()

const videoElement = ref<HTMLVideoElement>()
const isStreaming = ref(false)
const wsConnected = ref(false)
const isConnecting = ref(false)
const error = ref('')
const currentInstruction = ref('')
const ws = ref<WebSocket | null>(null)

const livenessStatus = ref({
  blinks: 0,
  ear: 0,
  yaw: 0,
  spoof_prob: 1.0,
  bbox: null as number[] | null
})

// Buffer for spoof probability to handle fluctuations
const spoofBuffer = ref<number[]>([])
const MIN_SPOOF_BUFFER_SIZE = 3

let stream: MediaStream | null = null
let frameInterval: number | null = null

// Helper function to check if spoof probability is consistently low
const isSpoofCheckPassed = computed(() => {
  if (spoofBuffer.value.length < MIN_SPOOF_BUFFER_SIZE) {
    return false
  }

  // Average of recent spoof probabilities should be < 0.7 (more lenient)
  const recentSpoofs = spoofBuffer.value.slice(-MIN_SPOOF_BUFFER_SIZE)
  const avgSpoof = recentSpoofs.reduce((sum, val) => sum + val, 0) / recentSpoofs.length

  return avgSpoof < 0.7 // More lenient threshold
})

const isLivenessComplete = computed(() => {
  const blinks = livenessStatus.value.blinks || 0
  const yaw = Math.abs(livenessStatus.value.yaw || 0)

  console.log('Liveness Debug:', {
    blinks,
    yaw,
    currentSpoof: livenessStatus.value.spoof_prob,
    avgSpoof: spoofBuffer.value.length >= MIN_SPOOF_BUFFER_SIZE
      ? spoofBuffer.value.slice(-MIN_SPOOF_BUFFER_SIZE).reduce((sum, val) => sum + val, 0) / MIN_SPOOF_BUFFER_SIZE
      : 'N/A',
    blinksCheck: blinks >= 2,
    yawCheck: yaw > 20,
    spoofCheck: isSpoofCheckPassed.value,
    bufferSize: spoofBuffer.value.length
  })

  return (
    blinks >= 2 &&
    yaw > 20 &&
    isSpoofCheckPassed.value
  )
})

onMounted(() => {
  connectWebSocket()
})

onUnmounted(() => {
  stopCamera()
  disconnectWebSocket()
})

const connectWebSocket = () => {
  isConnecting.value = true
  error.value = ''

  try {
    ws.value = new WebSocket('ws://localhost:8000/ws')

    ws.value.onopen = () => {
      wsConnected.value = true
      isConnecting.value = false
      console.log('WebSocket connected')
    }

    ws.value.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.error) {
        error.value = data.error
        return
      }

      currentInstruction.value = data.instruction || ''
      livenessStatus.value = {
        blinks: data.blinks_detected || 0,
        ear: data.ear || 0,
        yaw: data.yaw || 0,
        spoof_prob: data.spoof_prob || 1.0,
        bbox: data.bbox || null
      }

      // Update spoof buffer for more stable detection
      if (data.spoof_prob !== undefined) {
        spoofBuffer.value.push(data.spoof_prob)
        // Keep only last 10 values to prevent memory issues
        if (spoofBuffer.value.length > 10) {
          spoofBuffer.value = spoofBuffer.value.slice(-10)
        }
      }

      // Store liveness result when complete
      if (isLivenessComplete.value && !store.livenessResult) {
        console.log('Storing liveness result:', {
          blinks: livenessStatus.value.blinks,
          head_rotation: Math.abs(livenessStatus.value.yaw || 0),
          spoof_probability: livenessStatus.value.spoof_prob,
          yaw: livenessStatus.value.yaw
        })

        store.setLivenessResult({
          success: true,
          blinks: livenessStatus.value.blinks || 0,
          head_rotation: Math.abs(livenessStatus.value.yaw || 0),
          spoof_probability: livenessStatus.value.spoof_prob || 1,
          instruction: currentInstruction.value,
          completed_at: new Date().toISOString()
        })
      }
    }

    ws.value.onerror = (error) => {
      console.error('WebSocket error:', error)
      error.value = 'Failed to connect to liveness detection service'
      isConnecting.value = false
    }

    ws.value.onclose = () => {
      wsConnected.value = false
      console.log('WebSocket disconnected')
    }
  } catch (err) {
    error.value = 'Failed to establish WebSocket connection'
    isConnecting.value = false
  }
}

const disconnectWebSocket = () => {
  if (ws.value) {
    ws.value.close()
    ws.value = null
  }
}

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
      startFrameCapture()
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

  if (frameInterval) {
    clearInterval(frameInterval)
    frameInterval = null
  }

  isStreaming.value = false
}

const startFrameCapture = () => {
  if (!videoElement.value || !ws.value || ws.value.readyState !== WebSocket.OPEN) {
    return
  }

  frameInterval = window.setInterval(() => {
    captureAndSendFrame()
  }, 300) // Send frame every 300ms
}

const captureAndSendFrame = () => {
  if (!videoElement.value || !ws.value || ws.value.readyState !== WebSocket.OPEN) {
    return
  }

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  if (!ctx) return

  canvas.width = videoElement.value.videoWidth
  canvas.height = videoElement.value.videoHeight

  ctx.drawImage(videoElement.value, 0, 0)

  canvas.toBlob((blob) => {
    if (blob) {
      const reader = new FileReader()
      reader.onload = () => {
        const base64data = (reader.result as string).split(',')[1]

        ws.value?.send(JSON.stringify({
          frame: base64data
        }))
      }
      reader.readAsDataURL(blob)
    }
  }, 'image/jpeg', 0.8)
}

const restartProcess = () => {
  livenessStatus.value = {
    blinks: 0,
    ear: 0,
    yaw: 0,
    spoof_prob: 1.0,
    bbox: null
  }
  spoofBuffer.value = []
  currentInstruction.value = ''
  error.value = ''
}

const proceedToSelfie = () => {
  if (isLivenessComplete.value) {
    router.push('/selfie')
  }
}

const getInstructionIcon = (instruction: string) => {
  switch (instruction) {
    case 'blink': return '👁️'
    case 'turn_head': return '↺'
    case 'spoof_detected': return '⚠️'
    case 'done': return '✅'
    default: return '📷'
  }
}

const getInstructionText = (instruction: string) => {
  switch (instruction) {
    case 'blink': return 'Please Blink'
    case 'turn_head': return 'Turn Your Head'
    case 'spoof_detected': return 'Spoof Detected'
    case 'done': return 'Complete!'
    default: return 'Follow Instructions'
  }
}

const getInstructionDetail = (instruction: string) => {
  switch (instruction) {
    case 'blink': return 'Blink both eyes clearly'
    case 'turn_head': return 'Turn your head left or right'
    case 'spoof_detected': return 'Please ensure you are a live person'
    case 'done': return 'Liveness verification complete'
    default: return 'Position your face in the camera'
  }
}
</script>