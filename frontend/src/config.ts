declare global {
  interface Window {
    __APP_CONFIG__?: {
      apiBaseUrl?: string
    }
  }
}

export function getApiBaseUrl(): string {
  return (
    window.__APP_CONFIG__?.apiBaseUrl?.trim() ||
    import.meta.env.VITE_API_BASE_URL ||
    'http://localhost:8000'
  )
}
