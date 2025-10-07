<template>
  <div class="max-w-6xl mx-auto">
    <div class="card">
      <div class="text-center mb-8">
        <h2 class="text-3xl font-bold text-gray-900 mb-2">KYC Verification Results</h2>
        <p class="text-gray-600">Here's a summary of your identity verification process</p>
      </div>

      <!-- Overall Result -->
      <div class="text-center mb-12">
        <div
          class="inline-flex items-center justify-center w-24 h-24 rounded-full mb-4"
          :class="overallResult.success ? 'bg-green-100' : 'bg-red-100'"
        >
          <svg
            class="w-12 h-12"
            :class="overallResult.success ? 'text-green-600' : 'text-red-600'"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              v-if="overallResult.success"
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
            <path
              v-else
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </div>

        <h3 class="text-2xl font-bold mb-2" :class="overallResult.success ? 'text-green-800' : 'text-red-800'">
          {{ overallResult.success ? 'Verification Successful!' : 'Verification Failed' }}
        </h3>

        <p class="text-gray-600 mb-4">
          {{ overallResult.message }}
        </p>

        <div class="inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold"
             :class="overallResult.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'">
          {{ overallResult.confidence }}% Confidence
        </div>
      </div>

      <!-- Step Results -->
      <div class="grid gap-6 mb-8">
        <!-- License Validation Result -->
        <div class="border rounded-lg p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">License Validation</h3>
            <div class="flex items-center space-x-2">
              <div
                class="w-3 h-3 rounded-full"
                :class="licenseResult.success ? 'bg-green-400' : 'bg-red-400'"
              ></div>
              <span class="text-sm font-medium" :class="licenseResult.success ? 'text-green-600' : 'text-red-600'">
                {{ licenseResult.success ? 'Passed' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p class="font-medium text-gray-700">Authenticity Score</p>
              <p class="text-gray-600">{{ licenseResult.authenticity_score }}%</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Confidence Level</p>
              <p class="text-gray-600">{{ licenseResult.confidence_level }}</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">OCR Confidence</p>
              <p class="text-gray-600">{{ licenseResult.ocr_confidence }}%</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Processing Time</p>
              <p class="text-gray-600">{{ licenseResult.processing_time || 'N/A' }}</p>
            </div>
          </div>

          <!-- Validation Details -->
          <div class="mt-4 grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 rounded-full" :class="licenseResult.validations?.name_match ? 'bg-green-400' : 'bg-red-400'"></div>
              <span>Name Match</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 rounded-full" :class="licenseResult.validations?.dob_match ? 'bg-green-400' : 'bg-red-400'"></div>
              <span>DOB Match</span>
            </div>
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 rounded-full" :class="licenseResult.validations?.id_match ? 'bg-green-400' : 'bg-red-400'"></div>
              <span>ID Match</span>
            </div>
          </div>
        </div>

        <!-- Liveness Detection Result -->
        <div class="border rounded-lg p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">Liveness Detection</h3>
            <div class="flex items-center space-x-2">
              <div
                class="w-3 h-3 rounded-full"
                :class="livenessResult.success ? 'bg-green-400' : 'bg-red-400'"
              ></div>
              <span class="text-sm font-medium" :class="livenessResult.success ? 'text-green-600' : 'text-red-600'">
                {{ livenessResult.success ? 'Passed' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p class="font-medium text-gray-700">Blinks Detected</p>
              <p class="text-gray-600">{{ livenessResult.blinks || 0 }}/2</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Head Rotation</p>
              <p class="text-gray-600">{{ (livenessResult.head_rotation || 0).toFixed(1) }}°</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Spoof Probability</p>
              <p class="text-gray-600">{{ ((livenessResult.spoof_probability || 1) * 100).toFixed(1) }}%</p>
            </div>
          </div>
        </div>

        <!-- Selfie Validation Result -->
        <div class="border rounded-lg p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">Selfie Validation</h3>
            <div class="flex items-center space-x-2">
              <div
                class="w-3 h-3 rounded-full"
                :class="selfieResult?.match_result ? 'bg-green-400' : 'bg-red-400'"
              ></div>
              <span class="text-sm font-medium" :class="selfieResult?.match_result ? 'text-green-600' : 'text-red-600'">
                {{ selfieResult?.match_result ? 'Passed' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p class="font-medium text-gray-700">Similarity Score</p>
              <p class="text-gray-600">{{ selfieResult?.similarity_score?.toFixed(1) || 'N/A' }}%</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Confidence Level</p>
              <p class="text-gray-600">{{ selfieResult?.confidence_level || 'N/A' }}</p>
            </div>
            <div>
              <p class="font-medium text-gray-700">Match Threshold</p>
              <p class="text-gray-600">{{ selfieResult?.threshold_used || 'N/A' }}%</p>
            </div>
          </div>
        </div>
      </div>

      <!-- ID Information (if available) -->
      <div v-if="licenseResult.parsed_fields" class="border rounded-lg p-6 mb-8">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Extracted Information</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
          <div>
            <p class="font-medium text-gray-700">Name</p>
            <p class="text-gray-600">{{ getFullName(licenseResult.parsed_fields) }}</p>
          </div>
          <div>
            <p class="font-medium text-gray-700">Date of Birth</p>
            <p class="text-gray-600">{{ formatDate(licenseResult.parsed_fields.DBB) }}</p>
          </div>
          <div>
            <p class="font-medium text-gray-700">License Number</p>
            <p class="text-gray-600">{{ licenseResult.parsed_fields.DAQ || 'N/A' }}</p>
          </div>
          <div>
            <p class="font-medium text-gray-700">Expiration Date</p>
            <p class="text-gray-600">{{ formatDate(licenseResult.parsed_fields.DBA) }}</p>
          </div>
          <div>
            <p class="font-medium text-gray-700">Issue Date</p>
            <p class="text-gray-600">{{ formatDate(licenseResult.parsed_fields.DBD) }}</p>
          </div>
          <div>
            <p class="font-medium text-gray-700">State</p>
            <p class="text-gray-600">{{ licenseResult.parsed_fields.DCS || 'N/A' }}</p>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          @click="restartProcess"
          class="btn-secondary px-8 py-3"
        >
          Start New Verification
        </button>

        <button
          @click="downloadReport"
          class="btn-primary px-8 py-3"
        >
          Download Report
        </button>
      </div>

      <!-- Processing Information -->
      <div class="mt-8 text-center text-xs text-gray-500">
        <p>Verification completed on {{ new Date().toLocaleString() }}</p>
        <p class="mt-1">This verification is valid for 24 hours</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useKycStore } from '@/stores/kyc'

const router = useRouter()
const store = useKycStore()

const licenseResult = computed(() => store.licenseResult || {})
const livenessResult = computed(() => store.livenessResult || {})
const selfieResult = computed(() => store.selfieResult || {})

const overallResult = computed(() => {
  const licenseSuccess = licenseResult.value.success
  const livenessSuccess = livenessResult.value.success
  const selfieSuccess = selfieResult.value?.match_result

  const success = licenseSuccess && livenessSuccess && selfieSuccess

  let confidence = 0
  if (licenseSuccess && livenessSuccess && selfieSuccess) {
    const licenseScore = licenseResult.value.authenticity_score || 0
    const livenessScore = Math.min((livenessResult.value.blinks || 0) * 25 + (Math.abs(livenessResult.value.head_rotation || 0) / 20) * 25 + ((1 - (livenessResult.value.spoof_probability || 1)) * 100) * 0.5, 100)
    const selfieScore = selfieResult.value?.similarity_score || 0

    confidence = Math.round((licenseScore + livenessScore + selfieScore) / 3)
  }

  return {
    success,
    confidence,
    message: success
      ? 'Your identity has been successfully verified with high confidence.'
      : 'Verification failed. Please ensure all documents are valid and try again.'
  }
})

const getFullName = (fields: any) => {
  const nameParts = [
    fields.DAC, // First name
    fields.DAD, // Middle name
    fields.DCU  // Last name
  ].filter(Boolean)

  return nameParts.join(' ') || 'N/A'
}

const formatDate = (dateStr: string) => {
  if (!dateStr || dateStr.length !== 8) return 'N/A'

  try {
    // Assuming MMDDYYYY format
    const month = dateStr.substring(0, 2)
    const day = dateStr.substring(2, 4)
    const year = dateStr.substring(4, 8)

    return `${month}/${day}/${year}`
  } catch {
    return dateStr
  }
}

const restartProcess = () => {
  store.clearResults()
  router.push('/license')
}

const downloadReport = () => {
  const reportData = {
    timestamp: new Date().toISOString(),
    overallResult: overallResult.value,
    licenseValidation: licenseResult.value,
    livenessDetection: livenessResult.value,
    selfieValidation: selfieResult.value
  }

  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `kyc-verification-${new Date().toISOString().split('T')[0]}.json`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
</script>