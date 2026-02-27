/**
 * Vocal SDK - Unified voice synthesis and voice cloning SDK
 */

import axios, { AxiosInstance } from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import {
  VocalConfig,
  Voice,
  AudioResult,
  Result,
  CloneOptions,
  DesignOptions,
  SpeechOptions,
  MiniMaxConfig,
  MiniMaxUploadResponse,
  MiniMaxCloneResponse,
  MiniMaxT2AResponse,
  MiniMaxDesignResponse,
} from './types';

const DEFAULT_BASE_URL = 'https://api.minimaxi.com';

/**
 * Vocal SDK Main Class
 * Provides voice cloning, voice design and speech synthesis
 */
export class Vocal {
  private apiKey: string;
  private baseUrl: string;
  private client: AxiosInstance;

  constructor(config: VocalConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || DEFAULT_BASE_URL;

    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
      },
    });
  }

  /**
   * Upload audio file
   * @internal
   */
  private async uploadAudio(audioPath: string, purpose: string): Promise<string> {
    if (!fs.existsSync(audioPath)) {
      throw new Error(`Audio file not found: ${audioPath}`);
    }

    const fileBuffer = fs.readFileSync(audioPath);
    const filename = path.basename(audioPath);

    const formData = new FormData();
    formData.append('purpose', purpose);
    formData.append('file', new Blob([fileBuffer]), filename);

    try {
      const response = await this.client.post<MiniMaxUploadResponse>(
        '/v1/files/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      return response.data.file.file_id;
    } catch (error: any) {
      throw new Error(`Failed to upload audio: ${error.message}`);
    }
  }

  /**
   * Upload audio for cloning
   */
  async uploadCloneAudio(audioPath: string): Promise<string> {
    return this.uploadAudio(audioPath, 'voice_clone');
  }

  /**
   * Upload prompt audio
   */
  async uploadPromptAudio(audioPath: string): Promise<string> {
    return this.uploadAudio(audioPath, 'prompt_audio');
  }

  /**
   * Clone voice
   * @param options Clone options
   * @returns Clone result
   */
  async cloneVoice(options: CloneOptions): Promise<Result<Voice>> {
    try {
      // 1. Upload audio to clone
      const fileId = await this.uploadCloneAudio(options.audioPath);

      // 2. (Optional) Upload prompt audio
      let promptFileId: string | undefined;
      if (options.promptAudioPath) {
        promptFileId = await this.uploadPromptAudio(options.promptAudioPath);
      }

      // 3. Call clone API
      const request: any = {
        file_id: fileId,
        voice_id: options.voiceId,
        model: options.model || 'speech-2.8-hd',
        text: options.testText || '',
        language_boost: 'auto',
        need_volume_normalization: true,
        aigc_watermark: false,
      };

      if (promptFileId && options.promptText) {
        request.clone_prompt = {
          prompt_audio: promptFileId,
          prompt_text: options.promptText,
        };
      }

      const response = await this.client.post<MiniMaxCloneResponse>(
        '/v1/voice_clone',
        request,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const data = response.data;

      // Check for errors
      if (data.base_resp && data.base_resp.status_code !== 0) {
        return {
          success: false,
          error: data.base_resp.status_msg,
        };
      }

      // Convert to generic format
      return {
        success: true,
        data: {
          id: options.voiceId,
          previewUrl: data.demo_audio || undefined,
        },
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Simplified voice clone
   */
  async clone(
    audioPath: string,
    voiceId: string,
    testText?: string,
    model?: string
  ): Promise<Result<Voice>> {
    return this.cloneVoice({
      audioPath,
      voiceId,
      testText,
      model,
    });
  }

  /**
   * Voice design - Generate voice from text description
   */
  async designVoice(options: DesignOptions): Promise<Result<Voice>> {
    try {
      const request = {
        prompt: options.prompt,
        preview_text: options.previewText,
        voice_id: options.voiceId,
        aigc_watermark: options.watermark || false,
      };

      const response = await this.client.post<MiniMaxDesignResponse>(
        '/v1/voice_design',
        request,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const data = response.data;

      // Check for errors
      if (data.base_resp && data.base_resp.status_code !== 0) {
        return {
          success: false,
          error: data.base_resp.status_msg,
        };
      }

      return {
        success: true,
        data: {
          id: data.voice_id,
          previewAudio: data.trial_audio,
        },
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Simplified voice design
   */
  async design(
    prompt: string,
    previewText: string,
    voiceId?: string
  ): Promise<Result<Voice>> {
    return this.designVoice({
      prompt,
      previewText,
      voiceId,
    });
  }

  /**
   * Speech synthesis
   */
  async speech(options: SpeechOptions): Promise<Result<AudioResult>> {
    try {
      const request = {
        text: options.text,
        voice_id: options.voiceId,
        model: options.model || 'speech-2.8-hd',
        speed: options.speed,
        vol: options.volume,
        pitch: options.pitch,
        format: options.format,
        bitrate: options.bitrate,
        sample_rate: options.sampleRate,
      };

      const response = await this.client.post<MiniMaxT2AResponse>(
        '/v1/t2a',
        request,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      const data = response.data;

      return {
        success: true,
        data: {
          fileId: data.data.file_id,
          url: data.data.url,
        },
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Simplified speech synthesis
   */
  async speak(
    text: string,
    voiceId: string,
    model?: string
  ): Promise<Result<AudioResult>> {
    return this.speech({
      text,
      voiceId,
      model,
    });
  }
}

/**
 * Create Vocal instance
 */
export function createVocal(apiKey: string, baseUrl?: string): Vocal {
  return new Vocal({ apiKey, baseUrl });
}

// Legacy exports for compatibility
export { Vocal as MiniMaxVoiceClone, createVocal as createMiniMaxVoiceClone };

export default Vocal;
