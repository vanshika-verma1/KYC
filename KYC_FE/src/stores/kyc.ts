import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useKycStore = defineStore('kyc', () => {
  // State
  const licenseResult = ref<any>(null)
  const livenessResult = ref<any>(null)
  const selfieResult = ref<any>(null)
  const frontImageFile = ref<File | null>(null)

  // Getters
  const isComplete = computed(() => {
    return licenseResult.value && livenessResult.value && selfieResult.value
  })

  const overallStatus = computed(() => {
    if (!isComplete.value) return 'in_progress'

    const licenseSuccess = licenseResult.value.success
    const livenessSuccess = livenessResult.value.success
    const selfieSuccess = selfieResult.value?.match_result

    if (licenseSuccess && livenessSuccess && selfieSuccess) return 'success'
    return 'failed'
  })

  // Actions
  const setLicenseResult = (result: any) => {
    licenseResult.value = result
  }

  const setLivenessResult = (result: any) => {
    livenessResult.value = result
  }

  const setSelfieResult = (result: any) => {
    selfieResult.value = result
  }

  const setFrontImageFile = (file: File) => {
    frontImageFile.value = file
  }

  const clearResults = () => {
    licenseResult.value = null
    livenessResult.value = null
    selfieResult.value = null
    frontImageFile.value = null
  }

  return {
    // State
    licenseResult,
    livenessResult,
    selfieResult,
    frontImageFile,

    // Getters
    isComplete,
    overallStatus,

    // Actions
    setLicenseResult,
    setLivenessResult,
    setSelfieResult,
    setFrontImageFile,
    clearResults
  }
})