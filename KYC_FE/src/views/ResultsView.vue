<template>
  <div class="w-full">
    <!-- Incomplete Verification Notice -->
    <div v-if="!store.isComplete" class="fixed top-4 left-2 right-2 z-50 mb-6 p-5 bg-yellow-50/80 backdrop-blur-sm border border-yellow-200 rounded-2xl">
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

    <div class="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 p-6 mx-2 pt-10">
      <div class="text-center mb-4">
        <h2 class="text-3xl font-semibold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-1">
          KYC Verification Results
        </h2>
        <p class="text-gray-600 text-sm">Here's a summary of your identity verification process</p>
      </div>

      <!-- Enhanced Overall Result -->
      <div class="text-center">
        <div class="relative">
          <div
            class="inline-flex items-center justify-center w-24 h-24 rounded-full mb-4 relative overflow-hidden"
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
        </div>

        <h3 class="text-2xl font-bold mb-1" :class="overallResult.success ? 'text-green-800' : 'text-red-800'">
          {{ overallResult.success ? 'Verification Successful!' : 'Verification Failed' }}
        </h3>

        <p class="text-gray-600 mb-6 text-sm max-w-2xl mx-auto">
          {{ overallResult.message }}
        </p>
      </div>
 <!-- ID Information (if available) -->
      <div v-if="licenseResult.parsed_fields" class="bg-white/60 backdrop-blur-sm border border-white/30 rounded-2xl p-6 shadow-lg mb-4">
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
            <p class="text-gray-600">{{ licenseResult.parsed_fields.DAJ || 'N/A' }}</p>
          </div>
        </div>
      </div>
      <!-- Enhanced Step Results -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">

        <!-- License Validation Result -->
        <div class="bg-white/60 backdrop-blur-sm border border-white/30 rounded-2xl p-6 shadow-lg">
          <div class="flex items-center justify-between mb-6">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 class="text-l font-semibold text-gray-900">License Validation</h3>
            </div>
            <div class="flex items-center space-x-3">
              <span class="text-md font-semibold" :class="licenseResult.success ? 'text-green-600' : 'text-red-600'">
                {{ licenseResult.success ? 'Verified' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
            <div class="bg-white/50 rounded-xl p-4 text-center">
              <p class="font-semibold text-xl text-gray-800 mb-1">{{ licenseResult.authenticity_score*100 }}%</p>
              <p class="text-sm text-gray-600">Authenticity Score</p>
            </div>
            <div class="bg-white/50 rounded-xl p-4 text-center">
              <p class="font-semibold text-xl text-gray-800 mb-1">{{ licenseResult.confidence_level }}</p>
              <p class="text-sm text-gray-600">Confidence Level</p>
            </div>
          </div>

          <!-- Enhanced Validation Details -->
          <div class="bg-white/30 rounded-xl p-2">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full shadow-sm" :class="licenseResult.validations?.name_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-sm text-gray-700">Name Match</span>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full shadow-sm" :class="licenseResult.validations?.dob_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-sm text-gray-700">DOB Match</span>
              </div>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full shadow-sm" :class="licenseResult.validations?.id_match ? 'bg-green-400' : 'bg-red-400'"></div>
                <span class="font-medium text-sm text-gray-700">ID Match</span>
              </div>
            </div>
          </div>
        </div>


        <!-- Selfie Validation Result -->
        <div class="bg-white/60 backdrop-blur-sm border border-white/30 rounded-2xl p-6 shadow-lg">
          <div class="flex items-center justify-between mb-6">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <h3 class="text-l font-semibold text-gray-900">Selfie Validation</h3>
            </div>
            <div class="flex items-center space-x-3">
              <span class="text-md font-semibold" :class="selfieResult?.match_result ? 'text-green-600' : 'text-red-600'">
                {{ selfieResult?.match_result ? 'Verified' : 'Failed' }}
              </span>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-white/50 rounded-l p-4 text-center">
              <p class="font-semibold text-xl text-gray-800 mb-2">{{ selfieResult?.similarity_score?.toFixed(1) || 'N/A' }}%</p>
              <p class="text-sm text-gray-600">Similarity Score</p>
            </div>
            <div class="bg-white/50 rounded-l p-4 text-center">
              <p class="font-semibold text-xl text-gray-800 mb-2">{{ selfieResult?.confidence_level || 'N/A' }}</p>
              <p class="text-sm text-gray-600">Confidence Level</p>
            </div>
          </div>
        </div>
      </div>

     

      <!-- Enhanced Actions -->
      <div class="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          @click="restartProcess"
          class="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all"
        >
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Start New Verification
        </button>

        <button
          @click="downloadReport"
          class="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white text-sm font-medium py-2 px-4 rounded-lg shadow-md hover:shadow-lg transition-all"
        >
          <svg class="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l4-4m-4 4l-4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Download Report
        </button>
      </div>

      <!-- Enhanced Processing Information -->
      <div class="text-center">
        <div class="bg-white/40 backdrop-blur-sm rounded-2xl p-4 border border-white/20">
          <div class="flex items-center justify-center space-x-1">
            <svg class="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p class="text-xs font-medium text-gray-700">Verification completed on {{ new Date().toLocaleString() }}</p>
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
    fields.DCU,
    fields.DAC, // First name
    fields.DAD, // Middle name
    fields.DCS  // Last name
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

  const htmlContent = generateReportHTML(reportData)

  const blob = new Blob([htmlContent], { type: 'text/html' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `kyc-verification-report-${new Date().toISOString().split('T')[0]}.html`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

const generateReportHTML = (data: any) => {
  const formatDate = (dateStr: string) => {
    if (!dateStr || dateStr.length !== 8) return 'N/A'
    try {
      const month = dateStr.substring(0, 2)
      const day = dateStr.substring(2, 4)
      const year = dateStr.substring(4, 8)
      return `${month}/${day}/${year}`
    } catch {
      return dateStr
    }
  }

  const getFullName = (fields: any) => {
    const nameParts = [
      fields.DCU,
      fields.DAC,
      fields.DAD,
      fields.DCS
    ].filter(Boolean)
    return nameParts.join(' ') || 'N/A'
  }

  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>KYC Verification Report</title>
      <style>
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          margin: 0;
          padding: 40px;
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          color: #333;
          line-height: 1.6;
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
          background: white;
          border-radius: 15px;
          box-shadow: 0 20px 40px rgba(0,0,0,0.1);
          overflow: hidden;
        }
        .header {
          background: linear-gradient(135deg, ${data.overallResult.success ? '#10b981, #059669' : '#ef4444, #dc2626'});
          color: white;
          padding: 40px;
          text-align: center;
        }
        .header h1 {
          margin: 0;
          font-size: 2.5em;
          font-weight: 300;
          text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .header .subtitle {
          margin: 10px 0 0 0;
          font-size: 1.1em;
          opacity: 0.9;
        }
        .timestamp {
          background: rgba(255,255,255,0.2);
          padding: 8px 16px;
          border-radius: 20px;
          display: inline-block;
          margin-top: 15px;
          font-size: 0.9em;
        }
        .section {
          padding: 30px;
          border-bottom: 1px solid #eee;
        }
        .section:last-child {
          border-bottom: none;
        }
        .section h2 {
          color: #2d3748;
          margin-top: 0;
          font-size: 1.5em;
          display: flex;
          align-items: center;
        }
        .section h2::before {
          content: '';
          width: 4px;
          height: 20px;
          background: ${data.overallResult.success ? '#10b981' : '#ef4444'};
          margin-right: 12px;
          border-radius: 2px;
        }
        .overall-result {
          text-align: center;
          padding: 30px;
          background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
          margin-bottom: 20px;
        }
        .status-badge {
          display: inline-flex;
          align-items: center;
          padding: 12px 24px;
          border-radius: 50px;
          font-weight: 600;
          font-size: 1.1em;
          margin-bottom: 15px;
          ${data.overallResult.success
            ? 'background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46;'
            : 'background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b;'
          }
        }
        .info-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
          margin: 20px 0;
        }
        .info-card {
          background: #f8fafc;
          padding: 20px;
          border-radius: 10px;
          border-left: 4px solid ${data.overallResult.success ? '#10b981' : '#ef4444'};
        }
        .info-card .label {
          font-weight: 600;
          color: #4a5568;
          margin-bottom: 5px;
        }
        .info-card .value {
          color: #2d3748;
          font-size: 1.1em;
        }
        .score-card {
          text-align: center;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          border-radius: 10px;
          margin: 10px 0;
        }
        .score-card .score {
          font-size: 2em;
          font-weight: 700;
          margin-bottom: 5px;
        }
        .validation-item {
          display: flex;
          align-items: center;
          margin: 8px 0;
          padding: 8px;
          border-radius: 5px;
        }
        .validation-item.pass {
          background: rgba(16, 185, 129, 0.1);
        }
        .validation-item.fail {
          background: rgba(239, 68, 68, 0.1);
        }
        .validation-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-right: 12px;
          ${data.overallResult.success ? 'background: #10b981;' : 'background: #ef4444;'}
        }
        .footer {
          text-align: center;
          padding: 20px;
          background: #f8fafc;
          color: #718096;
          font-size: 0.9em;
        }
        @media print {
          body { background: white; }
          .container { box-shadow: none; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>KYC Verification Report</h1>
          <div class="subtitle">Identity Verification Results</div>
          <div class="timestamp">
            Generated on ${new Date(data.timestamp).toLocaleString()}
          </div>
        </div>

        <div class="overall-result">
          <div class="status-badge">
            ${data.overallResult.success ? '✓ VERIFICATION SUCCESSFUL' : '✗ VERIFICATION FAILED'}
          </div>
          <div style="font-size: 1.1em; margin-bottom: 10px;">
            ${data.overallResult.message}
          </div>
        </div>

        ${data.licenseValidation?.parsed_fields ? `
        <div class="section">
          <h2>Extracted Information</h2>
          <div class="info-grid">
            <div class="info-card">
              <div class="label">Full Name</div>
              <div class="value">${getFullName(data.licenseValidation.parsed_fields)}</div>
            </div>
            <div class="info-card">
              <div class="label">Date of Birth</div>
              <div class="value">${formatDate(data.licenseValidation.parsed_fields.DBB)}</div>
            </div>
            <div class="info-card">
              <div class="label">License Number</div>
              <div class="value">${data.licenseValidation.parsed_fields.DAQ || 'N/A'}</div>
            </div>
            <div class="info-card">
              <div class="label">Expiration Date</div>
              <div class="value">${formatDate(data.licenseValidation.parsed_fields.DBA)}</div>
            </div>
            <div class="info-card">
              <div class="label">Issue Date</div>
              <div class="value">${formatDate(data.licenseValidation.parsed_fields.DBD)}</div>
            </div>
            <div class="info-card">
              <div class="label">State</div>
              <div class="value">${data.licenseValidation.parsed_fields.DAJ || 'N/A'}</div>
            </div>
          </div>
        </div>
        ` : ''}

        <div class="section">
          <h2>License Validation</h2>
          <div class="info-grid">
            <div class="score-card">
              <div class="score">${Math.round((data.licenseValidation?.authenticity_score || 0) * 100)}%</div>
              <div>Authenticity Score</div>
            </div>
            <div class="score-card">
              <div class="score">${data.licenseValidation?.confidence_level || 'N/A'}</div>
              <div>Confidence Level</div>
            </div>
          </div>
          ${data.licenseValidation?.validations ? `
          <div style="margin-top: 20px;">
            <h3 style="margin-bottom: 15px; color: #4a5568;">Validation Details</h3>
            <div class="validation-item ${data.licenseValidation.validations.name_match ? 'pass' : 'fail'}">
              <div class="validation-dot"></div>
              <span>Name Match: ${data.licenseValidation.validations.name_match ? 'PASS' : 'FAIL'}</span>
            </div>
            <div class="validation-item ${data.licenseValidation.validations.dob_match ? 'pass' : 'fail'}">
              <div class="validation-dot"></div>
              <span>Date of Birth Match: ${data.licenseValidation.validations.dob_match ? 'PASS' : 'FAIL'}</span>
            </div>
            <div class="validation-item ${data.licenseValidation.validations.id_match ? 'pass' : 'fail'}">
              <div class="validation-dot"></div>
              <span>ID Match: ${data.licenseValidation.validations.id_match ? 'PASS' : 'FAIL'}</span>
            </div>
          </div>
          ` : ''}
        </div>

        ${data.selfieValidation ? `
        <div class="section">
          <h2>Selfie Validation</h2>
          <div class="info-grid">
            <div class="score-card">
              <div class="score">${data.selfieValidation.similarity_score?.toFixed(1) || 'N/A'}%</div>
              <div>Similarity Score</div>
            </div>
            <div class="score-card">
              <div class="score">${data.selfieValidation.confidence_level || 'N/A'}</div>
              <div>Confidence Level</div>
            </div>
          </div>
        </div>
        ` : ''}

        <div class="footer">
          <p>This report was generated automatically by the KYC Verification System.</p>
          <p>For security purposes, please verify all information before use.</p>
        </div>
      </div>
    </body>
    </html>
  `
}
</script>
