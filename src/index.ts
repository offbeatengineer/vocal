/**
 * Vocal SDK - 统一的语音合成和音色克隆 SDK
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
 * Vocal SDK 主类
 * 提供音色克隆、音色设计和语音合成功能
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
   * 上传音频文件
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
   * 上传待克隆的音频
   */
  async uploadCloneAudio(audioPath: string): Promise<string> {
    return this.uploadAudio(audioPath, 'voice_clone');
  }

  /**
   * 上传参考音频
   */
  async uploadPromptAudio(audioPath: string): Promise<string> {
    return this.uploadAudio(audioPath, 'prompt_audio');
  }

  /**
   * 音色克隆
   * @param options 克隆选项
   * @returns 克隆结果
   */
  async cloneVoice(options: CloneOptions): Promise<Result<Voice>> {
    try {
      // 1. 上传待克隆音频
      const fileId = await this.uploadCloneAudio(options.audioPath);

      // 2. (可选) 上传参考音频
      let promptFileId: string | undefined;
      if (options.promptAudioPath) {
        promptFileId = await this.uploadPromptAudio(options.promptAudioPath);
      }

      // 3. 调用克隆接口
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

      // 检查错误
      if (data.base_resp && data.base_resp.status_code !== 0) {
        return {
          success: false,
          error: data.base_resp.status_msg,
        };
      }

      // 转换为通用格式
      return {
        success: true,
        data: {
          id: options.voiceId,
          previewUrl: data.demo_audio || undefined,
          // 注意: clone 接口返回的是 URL，不是 hex
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
   * 简化版音色克隆
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
   * 音色设计 - 通过文字描述生成音色
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

      // 检查错误
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
   * 简化版音色设计
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
   * 语音合成
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
   * 简化版语音合成
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
 * 创建 Vocal 实例
 */
export function createVocal(apiKey: string, baseUrl?: string): Vocal {
  return new Vocal({ apiKey, baseUrl });
}

// 兼容旧版导出
export { Vocal as MiniMaxVoiceClone, createVocal as createMiniMaxVoiceClone };

export default Vocal;
