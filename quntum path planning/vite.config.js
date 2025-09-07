import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API requests to your backend server
      '/upload_excel': 'http://localhost:5000',
      '/resolve_digipin': 'http://localhost:5000',
      '/solve_tsp': 'http://localhost:5000',
    }
  }
})