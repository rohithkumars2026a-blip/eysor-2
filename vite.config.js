import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 3000,
    open: true
  },
  build: {
    target: 'esnext',
    outDir: 'dist'
  },
  optimizeDeps: {
    include: ['@tensorflow/tfjs', '@mediapipe/face_mesh', '@mediapipe/camera_utils']
  }
});
