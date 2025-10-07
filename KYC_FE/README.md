# KYC Frontend - Vue.js Application

A modern Vue.js frontend for KYC (Know Your Customer) verification with three-step identity validation process.

## Features

- **Step 1: License Validation** - Upload front and back images of government-issued ID
- **Step 2: Liveness Detection** - Real-time webcam verification with WebSocket communication
- **Step 3: Selfie Validation** - Face comparison between ID photo and live selfie
- **Results Dashboard** - Comprehensive verification results with confidence scores
- **Responsive Design** - Built with Tailwind CSS for modern UI/UX

## Tech Stack

- **Vue 3** - Progressive JavaScript framework
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **Vue Router** - Client-side routing
- **Pinia** - State management
- **Tailwind CSS** - Utility-first CSS framework

## Project Structure

```
src/
├── components/          # Reusable components
├── views/              # Page components
│   ├── LicenseValidation.vue    # Step 1: ID upload
│   ├── LivenessDetection.vue    # Step 2: Liveness check
│   ├── SelfieValidation.vue     # Step 3: Face matching
│   └── ResultsView.vue          # Results dashboard
├── stores/             # Pinia stores
│   └── kyc.ts         # KYC state management
├── main.ts            # Application entry point
└── style.css          # Global styles with Tailwind
```

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Python backend running on `http://localhost:8000`

## Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:8080`

## Backend Integration

The frontend communicates with the Python FastAPI backend:

### API Endpoints
- `POST /license/validate_license` - License validation
- `POST /selfie/validate_selfie` - Face comparison
- `WebSocket /ws` - Real-time liveness detection

### Proxy Configuration
The Vite dev server includes proxy configuration for seamless API communication:

```typescript
server: {
  proxy: {
    '/license': { target: 'http://localhost:8000' },
    '/selfie': { target: 'http://localhost:8000' },
    '/ws': { target: 'ws://localhost:8000', ws: true }
  }
}
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Key Components

#### LicenseValidation.vue
- File upload for ID images (front/back)
- Form validation and error handling
- API integration for license validation

#### LivenessDetection.vue
- WebSocket connection for real-time communication
- Camera access and video streaming
- Real-time instruction display
- Liveness verification metrics

#### SelfieValidation.vue
- Camera capture for selfie
- Face detection and quality assessment
- Face comparison API integration
- Validation results display

#### ResultsView.vue
- Comprehensive results summary
- Individual step status display
- Report generation and download

## State Management

The application uses Pinia for state management:

```typescript
// Access KYC store
const store = useKycStore()

// Store results from each step
store.setLicenseResult(licenseData)
store.setLivenessResult(livenessData)
store.setSelfieResult(selfieData)
```

## Styling

The application uses Tailwind CSS with custom utilities:

- **Responsive design** - Mobile-first approach
- **Custom components** - Reusable UI elements
- **Consistent theming** - Primary color scheme
- **Step indicators** - Visual progress tracking

## Browser Support

- Modern browsers with camera support
- WebSocket support required
- HTTPS recommended for camera access

## Production Deployment

1. **Build the application:**
   ```bash
   npm run build
   ```

2. **Configure backend URLs** for production environment

3. **Deploy static files** to web server

4. **Ensure backend API** is accessible from frontend domain

## Contributing

1. Follow Vue 3 composition API patterns
2. Use TypeScript for type safety
3. Maintain responsive design principles
4. Add proper error handling
5. Update documentation for new features

## License

This project is part of the KYC verification system.