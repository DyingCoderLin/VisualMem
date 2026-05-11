/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_VISUALMEM_THEME?: string
}

declare module '*.svg' {
  const content: string
  export default content
}

declare module '*.png' {
  const content: string
  export default content
}

declare module '*.jpg' {
  const content: string
  export default content
}
