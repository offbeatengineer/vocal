/**
 * Vocal SDK Types
 * Generic voice and speech synthesis interfaces, not coupled with any specific provider
 */

// ========== Generic Interfaces ==========

/**
 * SDK Configuration
 */
export interface VocalConfig {
  apiKey: string;
  baseUrl?: string;
}

/**
 * Voice information
 */
export interface Voice {
  /**
   * Unique voice identifier
   */
  id: string;
  
  /**
   * Voice name
   */
  name?: string;
  
  /**
   * Preview audio URL
   */
  previewUrl?: string;
  
  /**
   * Preview audio (hex encoded)
   */
  previewAudio?: string;
}

/**
 * Audio synthesis result
 */
export interface AudioResult {
  /**
   * File unique identifier
   */
  fileId: string;
  
  /**
   * Audio file URL
   */
  url: string;
}

/**
 * Operation result
 */
export interface Result<T = void> {
  /**
   * Whether successful
   */
  success: boolean;
  
  /**
   * Result data
   */
  data?: T;
  
  /**
   * Error message
   */
  error?: string;
}

/**
 * Voice clone options
 */
export interface CloneOptions {
  /**
   * Audio file path to clone
   */
  audioPath: string;
  
  /**
   * Custom voice_id
   */
  voiceId: string;
  
  /**
   * Prompt audio path (enhance cloning effect)
   */
  promptAudioPath?: string;
  
  /**
   * Prompt audio text
   */
  promptText?: string;
  
  /**
   * Test text (generate preview audio)
   */
  testText?: string;
  
  /**
   * Model to use
   */
  model?: string;
}

/**
 * Voice design options
 */
export interface DesignOptions {
  /**
   * Voice description
   */
  prompt: string;
  
  /**
   * Preview audio text
   */
  previewText: string;
  
  /**
   * Custom voice_id
   */
  voiceId?: string;
  
  /**
   * Whether to add watermark
   */
  watermark?: boolean;
}

/**
 * Speech synthesis options
 */
export interface SpeechOptions {
  /**
   * Text to synthesize
   */
  text: string;
  
  /**
   * Voice ID
   */
  voiceId: string;
  
  /**
   * Model to use
   */
  model?: string;
  
  /**
   * Speech speed
   */
  speed?: number;
  
  /**
   * Volume
   */
  volume?: number;
  
  /**
   * Pitch
   */
  pitch?: number;
  
  /**
   * Output format
   */
  format?: string;
  
  /**
   * Bitrate
   */
  bitrate?: number;
  
  /**
   * Sample rate
   */
  sampleRate?: number;
}

// ========== MiniMax Specific Types (Internal) ==========

/** @internal */
export interface MiniMaxConfig {
  apiKey: string;
  baseUrl?: string;
}

/** @internal */
export interface MiniMaxUploadResponse {
  file: {
    file_id: string;
    purpose: string;
    filename: string;
    size: number;
    created_at: number;
  };
}

/** @internal */
export interface MiniMaxCloneResponse {
  voice_id?: string;
  demo_audio?: string;
  audio?: {
    file_id: string;
    url: string;
  };
  model?: string;
  base_resp?: {
    status_code: number;
    status_msg: string;
  };
}

/** @internal */
export interface MiniMaxT2AResponse {
  data: {
    file_id: string;
    url: string;
  };
}

/** @internal */
export interface MiniMaxDesignResponse {
  voice_id: string;
  trial_audio: string;
  base_resp: {
    status_code: number;
    status_msg: string;
  };
}
