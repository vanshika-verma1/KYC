<template>
  <div class="max-w-7xl mx-auto">
    <!-- Incomplete Verification Notice -->
    <div v-if="!store.isComplete" class="mb-8 p-6 bg-yellow-50/80 backdrop-blur-sm border border-yellow-200 rounded-2xl">
      <div class="flex items-start">
        <div class="flex-shrink-0">
          <svg class="w-6 h-6 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
          </svg>
        </div>
        <div class="ml-3 flex-1">
          <h3 class="text-sm font-semibold text-yellow-800">Verification Incomplete</h3>
          <p class="mt-1 text-sm text-yellow-700 mb-3">
            You haven't completed all verification steps yet. Please complete the required steps to see your results.
          </p>
          <div class="flex flex-wrap gap-3">
            <button
              v-if="!licenseResult"
              @click="$router.push('/license')"
              class="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Complete License Validation
            </button>
            <button
              v-if="!selfieResult"
              @click="$router.push('/selfie')"
              class="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              Complete Selfie Validation
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-8">
      <div class="text-center mb-12">
        <div class="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
          <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 class="text-4xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-3">
          KYC Verification Results
        </h2>
        <p class="text-gray-600 text-lg">Here's a summary of your identity verification process</p>
      </div>

      <!-- Enhanced Overall Result -->
      <div class="text-center mb-12">
        <div class="relative mb-8">
          <div
            class="inline-flex items-center justify-center w-32 h-32 rounded-full mb-6 relative overflow-hidden"
            :class="overallResult.success ? 'bg-gradient-to-br from-green-100 to-emerald-100' : 'bg-gradient-to-br from-red-100 to-rose-100'"
          >
            <div class="absolute inset-0 bg-white/20 backdrop-blur-sm"></div>
            <svg
              class="w-16 h-16 relative z-10"
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
          <div class="absolute -top-2 -right-2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg" v-if="overallResult.success">
            <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
            </svg>
          </div>
        </div>

        <h3 class="text-3xl font-bold mb-4" :class="overallResult.success ? 'text-green-800' : 'text-red-800'">
          {{ overallResult.success ? 'Verification Successful!' : 'Verification Failed' }}
        </h3>

        <p class="text-gray-600 mb-6 text-lg max-w-2xl mx-auto">
          {{ overallResult.message }}
        </p>
      </div>

      <!-- Enhanced Step Results -->
      <div class="grid gap-8 mb-12">

        <!-- License Validation Result -->
        <div class="bg-white/60 backdrop-blur-sm border border-white/30 rounded-2xl p-8 shadow-lg">
          <div class="flex items-center justify-between mb-6">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 class="text-xl font-bold text-gray-900">License Validation</h3>
            </div>
            <div class="flex items-center space-x-3">
              <div
                class="w-4 h-4 rounded-full shadow-lg"
                :class="licenseResult.success ? 'bg-green-400' : 'bg-red-400'"
              ></div>
              <span class="text-lg font-semibold" :class="licenseResult.success ? 'text-green-600' : 'text-red-600'">
                {{ licenseResult.success ? 'Verified' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div class="bg-white/50 rounded-xl p-4 text-center">
              <p class="font-bold text-2xl text-gray-800 mb-1">{{ licenseResult.authenticity_score*100 }}%</p>
              <p class="text-sm text-gray-600">Authenticity Score</p>
            </div>
            <div class="bg-white/50 rounded-xl p-4 text-center">
              <p class="font-bold text-lg text-gray-800 mb-1">{{ licenseResult.confidence_level }}</p>
              <p class="text-sm text-gray-600">Confidence Level</p>
            </div>
          </div>

          <!-- Enhanced Validation Details -->
          <div class="bg-white/30 rounded-xl p-4">
            <h4 class="font-semibold text-gray-800 mb-3">Validation Details</h4>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full shadow-sm" :class="licenseResult.validations?.name_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-gray-700">Name Match</span>
              </div>
              <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full shadow-sm" :class="licenseResult.validations?.dob_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-gray-700">DOB Match</span>
              </div>
              <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full shadow-sm" :class="licenseResult.validations?.id_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-gray-700">ID Match</span>
              </div>
            </div>
          </div>
        </div>


        <!-- Selfie Validation Result -->
        <div class="bg-white/60 backdrop-blur-sm border border-white/30 rounded-2xl p-8 shadow-lg">
          <div class="flex items-center justify-between mb-6">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 class="text-xl font-bold text-gray-900">Selfie Validation</h3>
            </div>
            <div class="flex items-center space-x-3">
              <div
                class="w-4 h-4 rounded-full shadow-lg"
                :class="selfieResult?.match_result ? 'bg-green-400' : 'bg-red-400'"
              ></div>
              <span class="text-lg font-semibold" :class="selfieResult?.match_result ? 'text-green-600' : 'text-red-600'">
                {{ selfieResult?.match_result ? 'Verified' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-white/50 rounded-xl p-6 text-center">
              <p class="font-bold text-3xl text-gray-800 mb-2">{{ selfieResult?.similarity_score?.toFixed(1) || 'N/A' }}%</p>
              <p class="text-sm text-gray-600">Similarity Score</p>
            </div>
            <div class="bg-white/50 rounded-xl p-6 text-center">
              <p class="font-bold text-xl text-gray-800 mb-2">{{ selfieResult?.confidence_level || 'N/A' }}</p>
              <p class="text-sm text-gray-600">Confidence Level</p>
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

      <!-- Enhanced Actions -->
      <div class="flex flex-col sm:flex-row gap-6 justify-center">
        <button
          @click="restartProcess"
          class="bg-gradient-to-r from-gray-100 to-gray-200 hover:from-gray-200 hover:to-gray-300 text-gray-700 font-semibold py-4 px-10 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
        >
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Start New Verification
        </button>

        <button
          @click="downloadReport"
          class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-4 px-10 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:scale-105"
        >
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l4-4m-4 4l-4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Download Report
        </button>
      </div>

      <!-- Enhanced Processing Information -->
      <div class="mt-12 text-center">
        <div class="bg-white/40 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <div class="flex items-center justify-center space-x-2">
            <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p class="text-sm font-semibold text-gray-700">Verification completed on {{ new Date().toLocaleString() }}</p>
          </div>
        </div>
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
  const licenseSuccess = licenseResult.value?.success
  const selfieSuccess = selfieResult.value?.match_result

  const success = licenseSuccess && selfieSuccess

  let confidence = 0
  if (licenseSuccess && selfieSuccess) {
    const licenseScore = licenseResult.value?.authenticity_score || 0
    const selfieScore = selfieResult.value?.similarity_score || 0

    confidence = Math.round((licenseScore + selfieScore) / 2)
  }

  let message = 'Verification incomplete.'
  if (success) {
    message = 'Your identity has been successfully verified with high confidence.'
  } else if (licenseSuccess === false) {
    message = 'License verification failed. Please check your ID document and try again.'
  } else if (selfieSuccess === false) {
    message = 'Selfie verification failed. Please ensure your face is clearly visible and try again.'
  } else if (!licenseResult.value || !selfieResult.value) {
    message = 'Please complete all verification steps before viewing results.'
  }

  return {
    success,
    confidence,
    message
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