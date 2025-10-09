/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, any>, Record<string, any>, any>
  export default component
}

declare module '*.vue' {
  import { type ComponentOptions } from 'vue'
  const component: ComponentOptions
  export default component
}